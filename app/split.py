# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# split.py — División train/validación/prueba agrupada por texto.

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from app.settings import SEMILLA

N_FOLDS = 20
FOLDS_PRUEBA = frozenset({0, 1, 2})
FOLDS_VAL = frozenset({3, 4, 5})
FOLD_ENTRENAMIENTO = -1


def dividir_splits(labels, groups):
    """Divide 70/15/15 por grupos y rescata al train las clases que queden solo fuera."""
    unicos, inverso = np.unique(groups, return_inverse=True)
    etiquetas = np.array([labels[inverso == i][0] for i in range(len(unicos))])

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    fold_de_grupo = np.empty(len(unicos), dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        for fold, (_, idx) in enumerate(sgkf.split(unicos, etiquetas, groups=unicos)):
            fold_de_grupo[idx] = fold

    entrenamiento = ~np.isin(fold_de_grupo, list(FOLDS_PRUEBA | FOLDS_VAL))
    for clase in np.unique(etiquetas):
        pos = np.where(etiquetas == clase)[0]
        if not entrenamiento[pos].any():
            fold_de_grupo[pos[0]] = FOLD_ENTRENAMIENTO

    fold_de_fila = fold_de_grupo[inverso]
    train = np.where(~np.isin(fold_de_fila, list(FOLDS_PRUEBA | FOLDS_VAL)))[0]
    val = np.where(np.isin(fold_de_fila, list(FOLDS_VAL)))[0]
    prueba = np.where(np.isin(fold_de_fila, list(FOLDS_PRUEBA)))[0]
    return train, val, prueba
