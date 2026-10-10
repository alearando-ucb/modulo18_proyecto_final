# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# main.py — API FastAPI para clasificar ítems de gasto.

import csv
import json
from datetime import datetime

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
    description="API para clasificar descripciones de gasto en partidas del COG",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    item: str = Field(..., min_length=1, max_length=API_MAX_TEXT)


class FeedbackRequest(BaseModel):
    item: str = Field(..., min_length=1, max_length=API_MAX_TEXT)
    partida_predicha: str = Field(..., min_length=1, max_length=10)
    partida_correcta: str = Field(..., min_length=1, max_length=10)
    usuario: str = Field(..., min_length=1, max_length=100)


@app.on_event("startup")
def startup():
    """Carga modelos al iniciar el servidor."""
    cargar_modelos()


@app.get("/api/health")
def health():
    """Estado del servicio."""
    return {"status": "ok", "modelos": ["dnn", "logreg"]}


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Clasifica un ítem de gasto y retorna top-k partidas."""
    texto = req.item.strip()
    if not texto:
        raise HTTPException(status_code=422, detail="El texto no puede estar vacio")

    resultado = predecir(texto, top_k=API_TOP_K, modelo="dnn")
    return resultado


@app.get("/api/metrics")
def metrics():
    """Retorna las metricas de evaluacion de los modelos."""
    if not METRICS.exists():
        raise HTTPException(status_code=404, detail="Metricas no encontradas")
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/partidas")
def partidas():
    """Retorna nombres cortos de partidas."""
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


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Registra una correccion del usuario para reentrenamiento futuro."""
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


@app.get("/")
def index():
    """Sirve el frontend."""
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
