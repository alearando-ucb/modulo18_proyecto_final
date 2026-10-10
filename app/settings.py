# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# settings.py — Configuraciones del clasificador (crece por fases)

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_OOV = ROOT / "data" / "oov"
ARTIFACTS = ROOT / "artifacts"

RUTA_DATOS = DATA_RAW / "DATOS ENTRENAMIENTO.csv"
DATA_SYNTHETIC = DATA_RAW / "DATOS SINTETICOS.csv"
RUTA_OOV = DATA_OOV / "DATOS OOV.csv"
CLASIFICADOR = DATA_RAW / "clasificador_presupuestario_2026_enriquecido.csv"

EMBEDDINGS = ARTIFACTS / "embeddings.npz"
MODELO_LOGREG = ARTIFACTS / "logreg.joblib"
MODELO_DNN = ARTIFACTS / "dnn.pt"
HISTORIAL_DNN = ARTIFACTS / "training_history.json"

SEMILLA = 42


def fijar_semillas():
    """Fija random, numpy y torch para reproducibilidad."""
    import random

    random.seed(SEMILLA)

    try:
        import numpy as np

        np.random.seed(SEMILLA)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(SEMILLA)
        torch.cuda.manual_seed_all(SEMILLA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

UMBRAL_MINIMO = 6

MODELO_SEMANTICO = "intfloat/multilingual-e5-large-instruct"
DIM_EMBEDDINGS = 1024

METRICA_PRINCIPAL = "macro_f1"
METRICA_SECUNDARIA = "error_grave"