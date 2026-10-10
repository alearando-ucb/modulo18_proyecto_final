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

GRID_C = [0.01, 0.1, 1.0, 10.0]


def cargar_datos():
    """Carga features, labels y groups desde embeddings.npz, mapeando partidas a índices."""
    data = np.load(EMBEDDINGS, allow_pickle=True)
    features = data["features"]
    labels_str = data["labels"].astype(str)
    groups = data["item_ids"]
    partidas = data["partidas"].astype(str)

    partida_to_idx = {p: i for i, p in enumerate(partidas)}
    labels = np.array([partida_to_idx[p] for p in labels_str], dtype=np.int64)

    return features, labels, groups, partidas


def split_80_20(features, labels, groups):
    """Divide 80/20 por grupos únicos, estratificado por label."""
    unique_groups = np.unique(groups)
    unique_labels = []
    label_to_group = {}
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        label_to_group[g] = labels[idx[0]]
        unique_labels.append(labels[idx[0]])

    unique_labels = np.array(unique_labels)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    train_idx = []
    test_idx = []
    for train_g, test_g in sgkf.split(unique_groups, unique_labels, groups=unique_groups):
        train_groups = unique_groups[train_g]
        test_groups = unique_groups[test_g]
        for g in train_groups:
            train_idx.extend(np.where(groups == g)[0])
        for g in test_groups:
            test_idx.extend(np.where(groups == g)[0])
        break

    return features[train_idx], labels[train_idx], features[test_idx], labels[test_idx]


def seleccionar_c(features_train, labels_train, groups_train):
    """Selecciona el mejor C con StratifiedGroupKFold(5)."""
    scores = {}
    for c in GRID_C:
        fold_scores = []
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
        for train_g, val_g in sgkf.split(features_train, labels_train, groups=groups_train):
            modelo = LogisticRegression(
                C=c, max_iter=2000, class_weight="balanced", random_state=SEMILLA
            )
            modelo.fit(features_train[train_g], labels_train[train_g])
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
    features, labels, groups, partidas = cargar_datos()
    print(f"  features: {features.shape}")
    print(f"  labels: {labels.shape}")

    print("Split 80/20...")
    X_train, y_train, X_test, y_test = split_80_20(features, labels, groups)
    groups_train = groups[: len(X_train)]
    print(f"  train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    print("Seleccionando C con validación cruzada...")
    mejor_c, grid_scores = seleccionar_c(X_train, y_train, groups_train)

    print(f"Entrenando modelo final con C={mejor_c}...")
    modelo = LogisticRegression(
        C=mejor_c, max_iter=2000, class_weight="balanced", random_state=SEMILLA
    )
    modelo.fit(X_train, y_train)

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
