# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# 02_features.py — Genera embeddings y features de similitud para el dataset de entrenamiento.

import csv

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from app.settings import (
    CLASIFICADOR,
    DATA_CLEAN,
    DIM_EMBEDDINGS,
    EMBEDDING_DESCRIPCIONES,
    MODELO_SEMANTICO,
    ROOT,
    fijar_semillas,
)

ARTIFACTS = ROOT / "artifacts"


def leer_clasificador(ruta) -> dict:
    """Lee el clasificador enriquecido y devuelve dict partida -> descripcion."""
    desc = {}
    with open(ruta, "r", encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        partida_idx = 0
        desc_idx = 1
        for row in reader:
            if len(row) < 2:
                continue
            partida = row[partida_idx].strip()
            descripcion = row[desc_idx].strip()
            if partida and descripcion:
                desc[partida] = descripcion
    return desc


def main():
    """Genera embeddings de items y descripciones, calcula scores y guarda artifacts."""
    fijar_semillas()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CLEAN / "DATOS.csv", sep=";", encoding="iso-8859-1")
    partidas_dataset = sorted(df["partida"].astype(str).unique())

    print(f"Partidas en dataset: {len(partidas_dataset)}")

    desc_completas = leer_clasificador(CLASIFICADOR)
    print(f"Partidas con descripcion en clasificador: {len(desc_completas)}")

    faltantes = [p for p in partidas_dataset if p not in desc_completas]
    if faltantes:
        print(f"AVISO: {len(faltantes)} partidas sin descripcion: {faltantes}")

    descripciones = []
    partidas_con_desc = []
    for p in partidas_dataset:
        if p in desc_completas:
            descripciones.append(desc_completas[p])
            partidas_con_desc.append(p)

    print(f"Partidas con descripcion usadas: {len(partidas_con_desc)}")

    print(f"Cargando modelo semantico: {MODELO_SEMANTICO}")
    modelo = SentenceTransformer(MODELO_SEMANTICO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = modelo.to(device)
    print(f"Device: {device}")

    print("Generando embeddings de descripciones...")
    emb_desc = modelo.encode(
        descripciones,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).to(device)

    items = df["item_limpio"].tolist()
    print(f"Generando embeddings de {len(items)} items...")
    emb_items = modelo.encode(
        items,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).to(device)

    print("Calculando scores de similitud...")
    scores = torch.matmul(emb_items, emb_desc.T)

    features = torch.cat([emb_items, scores], dim=1).to(torch.float32)

    total_features = features.shape[1]
    print(f"Features shape: {features.shape} ({DIM_EMBEDDINGS} + {scores.shape[1]} = {total_features})")

    labels = df["partida"].astype(str).values

    np.savez(
        ARTIFACTS / "embeddings.npz",
        features=features.cpu().numpy(),
        labels=labels,
        partidas=np.array(partidas_con_desc),
        item_ids=df["item_limpio"].values,
    )

    # emb_descripciones: embeddings de las descripciones oficiales del clasificador.
    # Archivo separado para que la API los cargue sin depender de los datos de entrenamiento.
    np.save(EMBEDDING_DESCRIPCIONES, emb_desc.cpu().numpy())

    print(f"Guardado en {ARTIFACTS / 'embeddings.npz'}")
    print(f"  - features: {features.shape}")
    print(f"  - labels: {labels.shape}")
    print(f"  - partidas con score: {len(partidas_con_desc)}")
    print(f"Guardado en {EMBEDDING_DESCRIPCIONES}")
    print(f"  - emb_descripciones: {emb_desc.shape}")


if __name__ == "__main__":
    main()
