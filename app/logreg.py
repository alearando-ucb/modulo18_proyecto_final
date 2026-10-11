# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# logreg.py — Regresión logística multinomial sobre features 1094.

import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from app.settings import EMBEDDINGS, MODELO_LOGREG, SEMILLA, fijar_semillas
from app.split import dividir_splits

GRID_C = [0.01, 0.1, 1.0, 10.0]


def cargar_datos():
    """Carga features, labels, groups y pesos desde embeddings.npz, mapeando partidas a índices."""
    data = np.load(EMBEDDINGS, allow_pickle=True)
    features = data["features"]
    labels_str = data["labels"].astype(str)
    groups = data["item_ids"]
    partidas = data["partidas"].astype(str)
    weights = (
        data["weights"]
        if "weights" in data.files
        else np.ones(len(features), dtype=np.float32)
    )

    partida_to_idx = {p: i for i, p in enumerate(partidas)}
    labels = np.array([partida_to_idx[p] for p in labels_str], dtype=np.int64)

    return features, labels, groups, partidas, weights


def seleccionar_c(features_train, labels_train, groups_train, weights_train):
    """Selecciona el mejor C con StratifiedGroupKFold(5)."""
    scores = {}
    for c in GRID_C:
        fold_scores = []
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
        for train_g, val_g in sgkf.split(features_train, labels_train, groups=groups_train):
            modelo = LogisticRegression(
                C=c, max_iter=2000, class_weight="balanced", random_state=SEMILLA
            )
            modelo.fit(
                features_train[train_g],
                labels_train[train_g],
                sample_weight=weights_train[train_g],
            )
            pred = modelo.predict(features_train[val_g])
            fold_scores.append(f1_score(labels_train[val_g], pred, average="macro"))
        scores[c] = np.mean(fold_scores)
        print(f"  C={c}: macro-F1={scores[c]:.4f}")

    mejor_c = max(scores, key=scores.get)
    print(f"Mejor C: {mejor_c} (macro-F1={scores[mejor_c]:.4f})")
    return mejor_c, scores


def main():
    """Entrena la regresión logística y guarda el modelo."""
    fijar_semillas()

    print("Cargando embeddings...")
    features, labels, groups, partidas, weights = cargar_datos()
    print(f"  features: {features.shape}")
    print(f"  labels: {labels.shape}")

    print("Split 70/15/15...")
    train_idx, val_idx, prueba_idx = dividir_splits(labels, groups, weights > 1.0)
    X_train, y_train = features[train_idx], labels[train_idx]
    groups_train = groups[train_idx]
    w_train = weights[train_idx]
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}, prueba: {len(prueba_idx)}")
    print(f"  correcciones en train con peso extra: {int((w_train > 1.0).sum())}")

    print("Seleccionando C con validación cruzada...")
    mejor_c, grid_scores = seleccionar_c(X_train, y_train, groups_train, w_train)

    print(f"Entrenando modelo final con C={mejor_c}...")
    modelo = LogisticRegression(
        C=mejor_c, max_iter=2000, class_weight="balanced", random_state=SEMILLA
    )
    modelo.fit(X_train, y_train, sample_weight=w_train)

    print(f"Guardando modelo en {MODELO_LOGREG}")
    joblib.dump(
        {
            "modelo": modelo,
            "mejor_c": mejor_c,
            "grid_scores": grid_scores,
            "partidas": partidas,
        },
        MODELO_LOGREG,
    )

    print("Listo.")


if __name__ == "__main__":
    main()
