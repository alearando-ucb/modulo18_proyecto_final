# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# dnn.py — Arquitectura de la red neuronal clasificadora.

import torch
import torch.nn as nn


class DNNPartidas(nn.Module):
    """Red neuronal con dos capas ocultas para clasificación de partidas."""

    def __init__(self, dim_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.dnn_head = nn.Sequential(
            nn.Linear(dim_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dnn_head(features.to(torch.float32))
