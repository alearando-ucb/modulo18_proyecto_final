# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# 03_train.py — Entrena la DNN con hiperparámetros del experimento de referencia.

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, TensorDataset

from app.dnn import DNNPartidas
from app.settings import (
    EMBEDDINGS,
    HISTORIAL_DNN,
    MODELO_DNN,
    PESO_CORRECCIONES,
    fijar_semillas,
)
from app.split import dividir_splits

# Hiperparámetros óptimos del experimento de referencia
LR = 2e-4
WEIGHT_DECAY = 0.05
EPOCHS = 40
BATCH_SIZE = 32
DROPOUT = 0.3
PACIENCIA = 6
WARMUP_RATIO = 0.1
GRAD_CLIP = 0.5
SWA_START = 10


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


def calcular_class_weights(labels_train, num_classes):
    """Calcula pesos balanceados solo con datos de entrenamiento."""
    clases_presentes = np.unique(labels_train)
    weights = compute_class_weight("balanced", classes=clases_presentes, y=labels_train)
    class_weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(clases_presentes, weights):
        class_weights[int(cls)] = w
    return torch.tensor(class_weights, dtype=torch.float)


def cosine_warmup(optimizer, total_steps):
    """Scheduler cosine con warmup del 10%."""
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step / max(warmup_steps, 1), 1.0)
        if step < warmup_steps
        else 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1))),
    )
    return scheduler


def entrenar(modelo, train_loader, val_features, val_labels, device, weights_tensor):
    """Entrena la DNN con early stopping y SWA sobre val_loss."""
    criterion_train = nn.CrossEntropyLoss(weight=weights_tensor, reduction="none")
    criterion_val = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = cosine_warmup(optimizer, total_steps)
    swa_model = AveragedModel(modelo)
    swa_activado = False

    mejor_val_loss = float("inf")
    epocas_sin_mejora = 0
    mejor_estado = None
    historial = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(EPOCHS):
        t0 = time.time()

        modelo.train()
        total_loss = 0.0
        for features, labels, pesos in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = modelo(features)
            loss = (criterion_train(logits, labels) * pesos.to(device)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        modelo.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            logits = modelo(torch.tensor(val_features, dtype=torch.float32).to(device))
            loss = criterion_val(logits, torch.tensor(val_labels).to(device))
            val_loss = loss.item()
            val_preds = logits.argmax(dim=1).cpu().numpy()

        train_loss = total_loss / len(train_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        elapsed = time.time() - t0

        historial["epoch"].append(epoch + 1)
        historial["train_loss"].append(round(train_loss, 4))
        historial["val_loss"].append(round(val_loss, 4))
        historial["val_f1"].append(round(val_f1, 4))

        print(
            f"  Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | {elapsed:.1f}s"
        )

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            epocas_sin_mejora = 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PACIENCIA:
                print(f"  Early stopping en epoch {epoch + 1}")
                break

        if epoch >= SWA_START:
            swa_model.update_parameters(modelo)
            swa_activado = True

    if swa_activado:
        modelo.load_state_dict(swa_model.module.state_dict())
        print(f"  Pesos SWA promediados desde epoch {SWA_START}")
    else:
        modelo.load_state_dict(mejor_estado)
        print(f"  SWA no alcanzado (early stop antes de epoch {SWA_START}), usando mejor epoch")
    return historial


def main():
    """Entrena la DNN y guarda el modelo entrenado."""
    fijar_semillas()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Cargando embeddings...")
    features, labels, groups, partidas, weights = cargar_datos()
    print(f"  features: {features.shape}")

    print("Split 70/15/15...")
    train_idx, val_idx, prueba_idx = dividir_splits(labels, groups, weights > 1.0)
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    w_train = weights[train_idx]
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}, prueba: {len(prueba_idx)}")
    print(f"  correcciones en train con peso {PESO_CORRECCIONES}: {int((w_train > 1.0).sum())}")

    num_classes = len(partidas)
    print(f"  clases: {num_classes}")

    weights_tensor = calcular_class_weights(y_train, num_classes).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(w_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    modelo = DNNPartidas(
        dim_features=X_train.shape[1],
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"  parámetros: {total_params:,}")

    print(f"Entrenando (LR={LR}, WD={WEIGHT_DECAY}, dropout={DROPOUT}, clip={GRAD_CLIP}, max_epochs={EPOCHS})...")
    historial = entrenar(modelo, train_loader, X_val, y_val, device, weights_tensor)

    print(f"Guardando modelo en {MODELO_DNN}")
    torch.save(
        {
            "state_dict": modelo.state_dict(),
            "dim_features": X_train.shape[1],
            "num_classes": num_classes,
            "partidas": partidas,
            "dropout": DROPOUT,
        },
        MODELO_DNN,
    )

    print(f"Guardando historial en {HISTORIAL_DNN}")
    with open(HISTORIAL_DNN, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    print("Listo.")


if __name__ == "__main__":
    main()
