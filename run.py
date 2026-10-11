# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# run.py — Orquestador: regenera los artefactos del pipeline y sirve la API.

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from app.settings import (
    API_HOST,
    API_PORT,
    DATA_CLEAN,
    EMBEDDINGS,
    EMBEDDING_DESCRIPCIONES,
    ERROR_REVIEW,
    HISTORIAL_DNN,
    METRICS,
    MODELO_DNN,
    MODELO_LOGREG,
    RUTA_OOV,
)

ROOT = Path(__file__).resolve().parent
DATOS = DATA_CLEAN / "DATOS.csv"

PASOS = [
    ("01_prep_data", "app.01_prep_data", (DATOS,)),
    ("02_features", "app.02_features", (EMBEDDINGS, EMBEDDING_DESCRIPCIONES)),
    ("logreg", "app.logreg", (MODELO_LOGREG,)),
    ("03_train", "app.03_train", (MODELO_DNN, HISTORIAL_DNN)),
    ("04_evaluate", "app.04_evaluate", (METRICS, ERROR_REVIEW)),
]


def calcular_hash(ruta) -> str:
    """SHA-256 del archivo, primeros 16 caracteres (mismo criterio que 04_evaluate)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hash_oov_semantico(ruta) -> str:
    """SHA-256 del contenido semántico del OOV (ignora finales de línea CRLF/LF)."""
    import pandas as pd

    df = pd.read_csv(ruta, sep=";", encoding="utf-8")
    canonico = "\n".join(
        f"{p};{i}" for p, i in zip(df["PARTIDA"].astype(str), df["ITEM"].astype(str))
    )
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]


def evaluar_consistencia():
    """Compara los hashes actuales de datos, embeddings y OOV con metrics.json."""
    try:
        with open(METRICS, "r", encoding="utf-8") as f:
            hashes = json.load(f).get("hashes", {})
    except (OSError, json.JSONDecodeError):
        return False, False, False
    datos_ok = DATOS.exists() and hashes.get("datos") == calcular_hash(DATOS)
    oov_ok = RUTA_OOV.exists() and hashes.get("oov") == hash_oov_semantico(RUTA_OOV)
    emb_ok = EMBEDDINGS.exists() and hashes.get("embeddings") == calcular_hash(EMBEDDINGS)
    return datos_ok, oov_ok, emb_ok


def construir_plan(rebuild, datos_ok, oov_ok, emb_ok):
    """Decide los pasos 2..5. Retorna lista de (nombre, modulo, correr, motivo)."""
    plan = []
    cadena = False
    eval_pend = False
    for nombre, modulo, salidas in PASOS[1:]:
        faltan = not all(s.exists() for s in salidas)
        if rebuild:
            correr, motivo = True, "rebuild forzado"
        elif modulo == "app.02_features":
            correr = faltan or not datos_ok or not emb_ok
            motivo = "faltan artefactos" if faltan else "datos o embeddings cambiaron"
        elif modulo == "app.04_evaluate":
            correr = faltan or eval_pend or not (datos_ok and oov_ok)
            if faltan:
                motivo = "faltan artefactos"
            elif eval_pend or not datos_ok:
                motivo = "modelos o datos regenerados"
            else:
                motivo = "OOV cambiado"
        else:
            correr = faltan or cadena or not datos_ok
            if faltan:
                motivo = "faltan artefactos"
            else:
                motivo = "embeddings regenerados"
        if not correr:
            motivo = "artefactos al dia"
        if correr:
            if modulo == "app.02_features":
                cadena = True
            if modulo in ("app.02_features", "app.logreg", "app.03_train"):
                eval_pend = True
        plan.append((nombre, modulo, correr, motivo))
    return plan


def ejecutar_modulo(modulo):
    """Ejecuta un modulo del pipeline con el mismo interprete."""
    resultado = subprocess.run([sys.executable, "-m", modulo], cwd=ROOT)
    if resultado.returncode != 0:
        raise SystemExit(f"El paso {modulo} fallo (codigo {resultado.returncode}).")


def main():
    """Orquesta el pipeline y opcionalmente sirve la API."""
    parser = argparse.ArgumentParser(
        description="Regenera los artefactos del clasificador y sirve la API."
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Regenera todos los artefactos sin chequeos.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra los pasos que se ejecutarian sin correrlos.")
    parser.add_argument("--no-serve", action="store_true",
                        help="Regenera y termina sin iniciar la API.")
    parser.add_argument("--port", type=int, default=API_PORT,
                        help=f"Puerto de la API (default {API_PORT}).")
    args = parser.parse_args()

    total = len(PASOS)
    inicio = time.perf_counter()

    if args.dry_run:
        datos_ok, oov_ok, emb_ok = evaluar_consistencia()
        plan = construir_plan(args.rebuild, datos_ok, oov_ok, emb_ok)
        print("=== Dry-run: pasos que se ejecutarian ===")
        print(f"[1/{total}] 01_prep_data - A EJECUTAR (siempre, instantaneo)")
        for i, (nombre, _, correr, motivo) in enumerate(plan, start=2):
            estado = "A EJECUTAR" if correr else "OMITIDO"
            print(f"[{i}/{total}] {nombre} - {estado} ({motivo})")
        print("\nNota: el plan se calcula sobre el DATOS.csv actual;")
        print("01_prep_data puede cambiarlo al ejecutar de verdad.")
        return

    print(f"[1/{total}] 01_prep_data - ejecutando...", flush=True)
    t0 = time.perf_counter()
    ejecutar_modulo("app.01_prep_data")
    print(f"[1/{total}] 01_prep_data - listo ({time.perf_counter() - t0:.1f}s)")

    datos_ok, oov_ok, emb_ok = evaluar_consistencia()
    plan = construir_plan(args.rebuild, datos_ok, oov_ok, emb_ok)

    for i, (nombre, modulo, correr, motivo) in enumerate(plan, start=2):
        if correr:
            print(f"[{i}/{total}] {nombre} - ejecutando ({motivo})...", flush=True)
            t0 = time.perf_counter()
            ejecutar_modulo(modulo)
            print(f"[{i}/{total}] {nombre} - listo ({time.perf_counter() - t0:.1f}s)")
        else:
            print(f"[{i}/{total}] {nombre} - omitido ({motivo})")

    transcurrido = time.perf_counter() - inicio
    print(f"\nPipeline completado en {transcurrido:.0f}s.")

    if args.no_serve:
        return
    print(f"\nAPI en http://{API_HOST}:{args.port}/  (Ctrl+C para detener)", flush=True)
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", API_HOST, "--port", str(args.port)],
        cwd=ROOT,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline detenido.")
