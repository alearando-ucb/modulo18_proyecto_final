# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# 01_prep_data.py — Limpia el CSV de partidas y genera el manifiesto.

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from app.settings import DATA_CLEAN, DATA_SYNTHETIC, RUTA_DATOS, UMBRAL_MINIMO


def limpiar_texto(texto: str) -> str:
    """Minúsculas, quita ruido institucional y espacios extras."""
    texto = texto.lower()
    texto = re.sub(r"cdl|fcbcb|revoluciones|jum|juaf", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def generar_hash(texto: str) -> str:
    """Genera hash SHA-256 para un texto."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def leer_real(ruta: Path) -> list[dict]:
    """Lee el CSV original de entrenamiento."""
    rows = []
    with open(ruta, "r", encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            rows.append({
                "item": row[0].strip(),
                "partida": row[1].strip(),
            })
    return rows


def leer_sintetico(ruta: Path) -> list[dict]:
    """Lee el CSV de datos sintéticos."""
    rows = []
    with open(ruta, "r", encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            rows.append({
                "item": row[1].strip(),
                "partida": row[0].strip(),
            })
    return rows


def main():
    """Combina real + sintético, limpia y genera manifiesto."""
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    rows_real = leer_real(RUTA_DATOS)
    rows_sintetico = leer_sintetico(DATA_SYNTHETIC)

    rows = rows_real + rows_sintetico

    for r in rows:
        r["item_limpio"] = limpiar_texto(r["item"])

    conteo = Counter(r["partida"] for r in rows)
    rows_filtrados = [r for r in rows if conteo[r["partida"]] >= UMBRAL_MINIMO]

    manifest = []
    for i, r in enumerate(rows_filtrados):
        h = generar_hash(r["item_limpio"] + "|" + r["partida"] + "|" + str(i))
        manifest.append({
            "id": h[:12],
            "group_id": r["item_limpio"],
            "split": "train",
            "label": r["partida"],
            "class_name": r["item_limpio"],
            "sha256": h,
        })

    df = pd.DataFrame(rows_filtrados)
    df.to_csv(DATA_CLEAN / "DATOS.csv", index=False, sep=";", encoding="iso-8859-1")

    with open(DATA_CLEAN / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Real: {len(rows_real)} filas")
    print(f"Sintetico: {len(rows_sintetico)} filas")
    print(f"Combinado: {len(rows)} filas")
    print(f"Tras umbral >= {UMBRAL_MINIMO}: {len(rows_filtrados)} filas")
    print(f"Clases unicas: {len(conteo)}")
    print(f"Manifest: {len(manifest)} entradas")


if __name__ == "__main__":
    main()