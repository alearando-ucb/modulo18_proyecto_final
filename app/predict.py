# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# predict.py — Inferencia de un ítem nuevo sobre los modelos entrenados.

import json
import re
import time

import joblib
import numpy as np
import torch

from app.dnn import DNNPartidas
from app.settings import (
    CLASIFICADOR,
    MODELO_DNN,
    MODELO_LOGREG,
    MODELO_SEMANTICO,
    PARTIDAS_NOMBRES,
)

RUIDO_INSTITUCIONAL = re.compile(r"\b(cdl|fcbcb|revoluciones|jum|juaf)\b")
ESPACIOS = re.compile(r"\s+")

_modelo_st = None
_modelo_dnn = None
_modelo_logreg = None
_partidas = None
_descripciones = None
_emb_desc = None
_nombres = None
_desc_completas = None


def limpiar_texto(texto: str) -> str:
    """Aplica la misma limpieza que 01_prep_data: minúsculas, ruido institucional, espacios."""
    texto = texto.lower().strip()
    texto = RUIDO_INSTITUCIONAL.sub(" ", texto)
    texto = ESPACIOS.sub(" ", texto)
    return texto


def cargar_modelos():
    """Carga modelos, partidas, descripciones y nombres en memoria (una sola vez)."""
    global _modelo_st, _modelo_dnn, _modelo_logreg, _partidas, _descripciones, _emb_desc, _nombres, _desc_completas

    from sentence_transformers import SentenceTransformer

    print("Cargando modelo semantico...")
    _modelo_st = SentenceTransformer(MODELO_SEMANTICO)

    print("Cargando DNN...")
    dnn_data = torch.load(MODELO_DNN, map_location="cpu", weights_only=False)
    _modelo_dnn = DNNPartidas(dnn_data["dim_features"], dnn_data["num_classes"], dnn_data["dropout"])
    _modelo_dnn.load_state_dict(dnn_data["state_dict"])
    _modelo_dnn.eval()

    print("Cargando logreg...")
    logreg_data = joblib.load(MODELO_LOGREG)
    _modelo_logreg = logreg_data["modelo"]
    _partidas = logreg_data["partidas"]

    print("Cargando descripciones del clasificador...")
    _desc_completas = {}
    with open(CLASIFICADOR, "r", encoding="utf-8-sig") as f:
        import csv as csv_mod
        reader = csv_mod.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) >= 2:
                _desc_completas[row[0].strip()] = row[1].strip()

    _descripciones = [_desc_completas.get(p, "") for p in _partidas]

    print("Pre-codificando descripciones...")
    _emb_desc = _modelo_st.encode(_descripciones, normalize_embeddings=True)

    print("Cargando nombres de partidas...")
    with open(PARTIDAS_NOMBRES, "r", encoding="utf-8") as f:
        _nombres = json.load(f)

    print("Modelos cargados.")


def generar_features(texto_limpio: str) -> np.ndarray:
    """Genera vector de features (embedding + scores) para un texto limpio."""
    emb_item = _modelo_st.encode([texto_limpio], normalize_embeddings=True)
    scores = emb_item @ _emb_desc.T
    features = np.concatenate([emb_item, scores], axis=1).astype(np.float32)
    return features


def predecir(texto: str, top_k: int = 3, modelo: str = "dnn") -> dict:
    """Predice la partida de un texto y retorna top-k con scores."""
    inicio = time.perf_counter()

    texto_limpio = limpiar_texto(texto)
    features = generar_features(texto_limpio)

    if modelo == "dnn":
        with torch.no_grad():
            logits = _modelo_dnn(torch.tensor(features))
            probs = torch.softmax(logits, dim=1).numpy()[0]
    else:
        probs = _modelo_logreg.predict_proba(features)[0]

    top_idx = probs.argsort()[::-1][:top_k]
    scores = [
        {
            "partida": _partidas[i],
            "nombre": _nombres.get(_partidas[i], ""),
            "score": round(float(probs[i]), 4),
        }
        for i in top_idx
    ]

    elapsed_ms = round((time.perf_counter() - inicio) * 1000, 1)

    return {
        "partida_predicha": scores[0]["partida"],
        "nombre_predicho": scores[0]["nombre"],
        "scores": scores,
        "confianza": scores[0]["score"],
        "elapsed_ms": elapsed_ms,
    }
