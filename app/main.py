# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# main.py — API FastAPI para clasificar ítems de gasto.

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.predict import cargar_modelos, predecir
from app.settings import API_MAX_TEXT, API_TOP_K, FRONTEND, METRICS, PARTIDAS_NOMBRES

app = FastAPI(
    title="Clasificador de Partidas de Gasto",
    description="API para clasificar descripciones de gasto en partidas del COG",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    item: str = Field(..., min_length=1, max_length=API_MAX_TEXT)


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


@app.get("/")
def index():
    """Sirve el frontend."""
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
