# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# main.py — API FastAPI para clasificar ítems de gasto.

import csv
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.predict import cargar_modelos, predecir
from app.settings import (
    API_MAX_TEXT,
    API_TOP_K,
    CLASIFICADOR,
    FRONTEND,
    METRICS,
    PARTIDAS_NOMBRES,
    RUTA_CORRECCIONES,
)

app = FastAPI(
    title="Clasificador de Partidas de Gasto",
    description=(
        "API para clasificar descripciones de ítems de gasto en las partidas del "
        "Clasificador por Objeto del Gasto (COG) del Ministerio de Economía y "
        "Finanzas Públicas de Bolivia.\n\n"
        "**Modelo por defecto:** DNN (ganador por Macro-F1 en prueba y OOV).\n\n"
        "**Scores:** provienen de un softmax sobre la salida de la DNN; no son "
        "probabilidades calibradas.\n\n"
        "**Correcciones:** el endpoint `/api/feedback` registra errores de "
        "predicción en `DATOS CORRECCIONES.csv`. Esas filas se incorporan al "
        "siguiente reentrenamiento con prioridad y peso 3× sobre el resto de los "
        "datos."
    ),
    version="1.2.0",
    contact={"name": "Arando Espejo Alejandro Marcelo"},
)


class ScoreItem(BaseModel):
    """Puntaje de una partida en el top-k de la predicción."""

    partida: str = Field(..., examples=["32100"], description="Código de la partida del COG")
    nombre: str = Field(..., examples=["Papel de oficina y parecidos"], description="Nombre corto de la partida")
    score: float = Field(..., examples=[0.6521], description="Puntaje softmax (0 a 1, no calibrado)")


class PredictRequest(BaseModel):
    item: str = Field(
        ...,
        min_length=1,
        max_length=API_MAX_TEXT,
        description=f"Descripción libre del ítem de gasto (1 a {API_MAX_TEXT} caracteres)",
        examples=["papel bond t/oficio para impresion institucional"],
    )


class PredictResponse(BaseModel):
    """Resultado de la clasificación de un ítem."""

    item: str = Field(..., description="Texto limpio recibido")
    partida_predicha: str = Field(..., examples=["32100"], description="Código de la partida predicha")
    nombre_predicho: str = Field(..., examples=["Papel de oficina y parecidos"], description="Nombre de la partida predicha")
    scores: List[ScoreItem] = Field(..., description=f"Top-{API_TOP_K} partidas ordenadas por puntaje descendente")
    confianza: float = Field(..., examples=[0.6521], description="Puntaje de la partida más probable")
    elapsed_ms: float = Field(..., examples=[95.2], description="Tiempo de inferencia en milisegundos")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    modelos: List[str] = Field(..., examples=[["dnn", "logreg"]])


class FeedbackRequest(BaseModel):
    item: str = Field(
        ...,
        min_length=1,
        max_length=API_MAX_TEXT,
        description=f"Descripción original del ítem de gasto (1 a {API_MAX_TEXT} caracteres)",
        examples=["monitor lcd de 27 pulgadas"],
    )
    partida_predicha: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Código de la partida que predijo el modelo (debe existir en el clasificador)",
        examples=["39800"],
    )
    partida_correcta: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Código de la partida correcta asignada por el usuario (debe existir en el clasificador)",
        examples=["43120"],
    )
    usuario: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Nombre o identificador de quien registra la corrección",
        examples=["alejandro arando"],
    )


class FeedbackResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    fecha: str = Field(..., examples=["2026-10-11 12:00:00"], description="Fecha/hora del registro")


@app.on_event("startup")
def startup():
    """Carga modelos al iniciar el servidor."""
    cargar_modelos()


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Evaluación"],
    summary="Estado del servicio",
)
def health():
    """Indica que la API está operativa y qué modelos tiene cargados."""
    return {"status": "ok", "modelos": ["dnn", "logreg"]}


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    tags=["Predicción"],
    summary="Clasifica un ítem de gasto",
    status_code=200,
    responses={
        422: {
            "description": "Texto vacío o fuera de los límites de longitud",
            "content": {
                "application/json": {
                    "example": {"detail": "El texto no puede estar vacio"}
                }
            },
        }
    },
)
def predict(req: PredictRequest):
    """Clasifica una descripción de gasto y retorna la partida predicha con el top-3 de puntajes.

    El modelo utilizado es la DNN (semilla seleccionada por validación). Los scores
    provienen de un softmax y representan la confianza relativa entre las partidas
    candidatas; no son probabilidades calibradas.
    """
    texto = req.item.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="El texto no puede estar vacio")

    resultado = predecir(texto, top_k=API_TOP_K, modelo="dnn")
    return resultado


@app.get(
    "/api/metrics",
    tags=["Evaluación"],
    summary="Métricas de evaluación",
    responses={
        200: {
            "description": (
                "JSON con las métricas completas de ambos modelos en prueba y OOV: "
                "Macro-F1, accuracy, balanced accuracy, error grave, log-loss, "
                "matriz de confusión y métricas por clase."
            )
        },
        404: {"description": "Archivo de métricas no encontrado (reentrenar el pipeline)"},
    },
)
def metrics():
    """Retorna las métricas de evaluación de ambos modelos (prueba y OOV).

    El payload es el archivo `artifacts/metrics.json` completo, generado por
    `04_evaluate.py` tras cada reentrenamiento.
    """
    if not METRICS.exists():
        raise HTTPException(status_code=404, detail="Metricas no encontradas")
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get(
    "/api/partidas",
    tags=["Catálogo"],
    summary="Catálogo de partidas",
    responses={
        200: {
            "description": (
                "Objeto JSON con todas las partidas del clasificador: código como "
                "clave y nombre corto como valor. Usado por el selector de "
                "correcciones del frontend."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "11100": "Haberes básicos",
                        "32100": "Papel de oficina y parecidos",
                        "43120": "Equipo de computación",
                    }
                }
            },
        },
        404: {"description": "Archivo de nombres de partidas no encontrado"},
    },
)
def partidas():
    """Retorna el catálogo completo de partidas (código y nombre corto)."""
    if not PARTIDAS_NOMBRES.exists():
        raise HTTPException(status_code=404, detail="Partidas no encontradas")
    with open(PARTIDAS_NOMBRES, "r", encoding="utf-8") as f:
        return json.load(f)


def _partidas_validas() -> set:
    """Lee los codigos de partida del clasificador oficial."""
    validas = set()
    with open(CLASIFICADOR, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if row and row[0].strip():
                validas.add(row[0].strip())
    return validas


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    tags=["Corrección"],
    summary="Registra una corrección de predicción",
    status_code=200,
    responses={
        422: {
            "description": (
                "Partida correcta o predicha no existe en el clasificador oficial, "
                "o algún campo está fuera de los límites de longitud."
            ),
            "content": {
                "application/json": {
                    "example": {"detail": "Partida correcta no existe en el clasificador"}
                }
            },
        }
    },
)
def feedback(req: FeedbackRequest):
    """Registra una corrección del usuario para reentrenamiento futuro.

    Las correcciones se acumulan en `data/raw/DATOS CORRECCIONES.csv` con prioridad
    sobre datos reales y sintéticos. Cada fila corregida se fuerza al split de
    entrenamiento y pesa 3× (`PESO_CORRECCIONES`) en el entrenamiento de ambos
    modelos. Los códigos de partida se validan contra el clasificador oficial
    antes de registrar.
    """
    validas = _partidas_validas()
    if req.partida_correcta not in validas:
        raise HTTPException(status_code=422, detail="Partida correcta no existe en el clasificador")
    if req.partida_predicha not in validas:
        raise HTTPException(status_code=422, detail="Partida predicha no existe en el clasificador")

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    item = req.item.strip().replace("\n", " ")

    with open(RUTA_CORRECCIONES, "a", encoding="iso-8859-1", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([req.partida_correcta, item, req.usuario.strip(), fecha, req.partida_predicha])

    return {"status": "ok", "fecha": fecha}


@app.get("/", include_in_schema=False)
def index():
    """Sirve el frontend."""
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
