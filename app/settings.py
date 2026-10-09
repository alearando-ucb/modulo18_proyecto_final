# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# settings.py — Configuraciones del clasificador (crece por fases)

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"

RUTA_DATOS = DATA_RAW / "DATOS ENTRENAMIENTO.csv"
DATA_SYNTHETIC = DATA_RAW / "DATOS SINTETICOS.csv"

SEMILLA = 42

UMBRAL_MINIMO = 6

MODELO_SEMANTICO = "intfloat/multilingual-e5-large-instruct"
DIM_EMBEDDINGS = 1024
TOTAL_FEATURES = 1084