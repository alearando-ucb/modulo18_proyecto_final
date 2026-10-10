# Autor: Arando Espejo Alejandro Marcelo
# Clasificador de Partidas de Gasto — AI Capstone Project
# 04_evaluate.py — Evalúa logreg y DNN en prueba y OOV, genera métricas y review de errores.

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from app.dnn import DNNPartidas
from app.settings import (
    CLASIFICADOR,
    DATA_CLEAN,
    DIM_EMBEDDINGS,
    EMBEDDINGS,
    ERROR_REVIEW,
    METRICS,
    MODELO_DNN,
    MODELO_LOGREG,
    MODELO_SEMANTICO,
    OOV_EMBEDDINGS,
    RUTA_OOV,
    SEMILLA,
    fijar_semillas,
)

# Pares de confusión graves congelados en .docs/gravedad.md
PARES_GRAVES = {
    ("3", "4"), ("4", "3"),
    ("2", "6"), ("6", "2"),
    ("1", "6"), ("6", "1"),
    ("2", "3"), ("3", "2"),
}


def calcular_hash(ruta) -> str:
    """Calcula hash SHA-256 de un archivo."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cargar_split():
    """Recupera el mismo split 80/20 usado en entrenamiento."""
    data = np.load(EMBEDDINGS, allow_pickle=True)
    features = data["features"]
    labels_str = data["labels"].astype(str)
    groups = data["item_ids"]
    partidas = data["partidas"].astype(str)

    partida_to_idx = {p: i for i, p in enumerate(partidas)}
    labels = np.array([partida_to_idx[p] for p in labels_str], dtype=np.int64)

    unique_groups = np.unique(groups)
    unique_labels = np.array([labels[groups == g][0] for g in unique_groups])

    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    train_g, test_g = next(sgkf.split(unique_groups, unique_labels, groups=unique_groups))

    test_groups = unique_groups[test_g]
    test_idx = np.where(np.isin(groups, test_groups))[0]

    return features[test_idx], labels[test_idx], partidas, groups[test_idx], labels_str[test_idx]


def tasa_error_grave(y_real, y_pred, partidas):
    """Calcula la tasa de error grave según los pares congelados."""
    errores = 0
    for r, p in zip(y_real, y_pred):
        grupo_real = partidas[r][0]
        grupo_pred = partidas[p][0]
        if (grupo_real, grupo_pred) in PARES_GRAVES:
            errores += 1
    return errores / len(y_real)


def evaluar_modelo(y_real, y_pred, y_prob, partidas, nombre):
    """Calcula todas las métricas para un modelo."""
    macro_f1 = f1_score(y_real, y_pred, average="macro")
    acc = accuracy_score(y_real, y_pred)
    bal_acc = balanced_accuracy_score(y_real, y_pred)
    error_grave = tasa_error_grave(y_real, y_pred, partidas)

    try:
        ll = log_loss(y_real, y_prob, labels=list(range(len(partidas))))
    except ValueError:
        ll = None

    precision, recall, f1, support = precision_recall_fscore_support(
        y_real, y_pred, labels=list(range(len(partidas))), zero_division=0
    )

    por_clase = {}
    for i, p in enumerate(partidas):
        if support[i] > 0:
            por_clase[p] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }

    cm = confusion_matrix(y_real, y_pred, labels=list(range(len(partidas))))

    return {
        "nombre": nombre,
        "macro_f1": round(float(macro_f1), 4),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "error_grave": round(float(error_grave), 4),
        "log_loss": round(float(ll), 4) if ll is not None else None,
        "por_clase": por_clase,
        "matriz_confusion": cm.tolist(),
    }


def generar_features_oov():
    """Genera embeddings y features para el conjunto OOV."""
    from sentence_transformers import SentenceTransformer

    oov = pd.read_csv(RUTA_OOV, sep=";", encoding="utf-8")
    items = oov["ITEM"].str.lower().str.strip().tolist()
    labels_str = oov["PARTIDA"].astype(str).tolist()

    data = np.load(EMBEDDINGS, allow_pickle=True)
    partidas_train = data["partidas"].astype(str)

    desc_completas = {}
    with open(CLASIFICADOR, "r", encoding="utf-8-sig") as f:
        reader = pd.read_csv(f, sep=";", encoding="utf-8-sig", header=None)
        for _, row in reader.iterrows():
            desc_completas[str(row[0]).strip()] = str(row[1]).strip()

    descripciones = [desc_completas.get(p, "") for p in partidas_train]

    print("Cargando modelo semántico...")
    modelo = SentenceTransformer(MODELO_SEMANTICO)

    print("Generando embeddings de descripciones...")
    emb_desc = modelo.encode(descripciones, normalize_embeddings=True, show_progress_bar=True)

    print(f"Generando embeddings de {len(items)} items OOV...")
    emb_items = modelo.encode(items, normalize_embeddings=True, show_progress_bar=True)

    scores = emb_items @ emb_desc.T
    features = np.concatenate([emb_items, scores], axis=1).astype(np.float32)

    np.savez(
        OOV_EMBEDDINGS,
        features=features,
        labels=np.array(labels_str),
        item_ids=np.array(items),
    )

    return features, labels_str, partidas_train


def review_errores(y_real, y_pred, partidas, labels_str, nombre):
    """Genera análisis de errores en markdown."""
    lineas = [f"## Review de errores — {nombre}\n"]

    errores = []
    for i, (r, p) in enumerate(zip(y_real, y_pred)):
        if r != p:
            grupo_real = partidas[r][0]
            grupo_pred = partidas[p][0]
            es_grave = (grupo_real, grupo_pred) in PARES_GRAVES
            errores.append({
                "item": labels_str[i],
                "real": partidas[r],
                "pred": partidas[p],
                "grave": es_grave,
            })

    graves = [e for e in errores if e["grave"]]
    lineas.append(f"- Total errores: {len(errores)} / {len(y_real)}")
    lineas.append(f"- Errores graves: {len(graves)}\n")

    confusions = {}
    for e in errores:
        key = (e["real"], e["pred"])
        confusions[key] = confusions.get(key, 0) + 1

    top = sorted(confusions.items(), key=lambda x: -x[1])[:15]
    lineas.append("### Top confusiones\n")
    lineas.append("| Real | Predicha | Frecuencia | Grave |")
    lineas.append("|------|----------|------------|-------|")
    for (real, pred), count in top:
        es_grave = "SI" if (real[0], pred[0]) in PARES_GRAVES else ""
        lineas.append(f"| {real} | {pred} | {count} | {es_grave} |")

    if graves:
        lineas.append("\n### Ejemplos de errores graves\n")
        for e in graves[:10]:
            lineas.append(f"- `{e['item'][:60]}` → real `{e['real']}`, predicha `{e['pred']}`")

    return "\n".join(lineas)


def main():
    """Ejecuta evaluación completa en prueba y OOV."""
    fijar_semillas()

    print("=== Evaluación en conjunto de prueba ===")
    X_test, y_test, partidas, groups_test, labels_str_test = cargar_split()
    print(f"  Tamaño de prueba: {X_test.shape[0]}")

    # Logreg
    print("\nLogreg:")
    logreg_data = joblib.load(MODELO_LOGREG)
    logreg_model = logreg_data["modelo"]
    logreg_pred = logreg_model.predict(X_test).astype(np.int64)
    logreg_prob = logreg_model.predict_proba(X_test)

    metricas_logreg = evaluar_modelo(y_test, logreg_pred, logreg_prob, partidas, "logreg")
    print(f"  macro-F1: {metricas_logreg['macro_f1']}")
    print(f"  accuracy: {metricas_logreg['accuracy']}")
    print(f"  error_grave: {metricas_logreg['error_grave']}")

    # DNN
    print("\nDNN:")
    dnn_data = torch.load(MODELO_DNN, map_location="cpu", weights_only=False)
    dnn_model = DNNPartidas(dnn_data["dim_features"], dnn_data["num_classes"], dnn_data["dropout"])
    dnn_model.load_state_dict(dnn_data["state_dict"])
    dnn_model.eval()

    with torch.no_grad():
        logits = dnn_model(torch.tensor(X_test, dtype=torch.float32))
        dnn_prob = torch.softmax(logits, dim=1).numpy()
        dnn_pred = dnn_prob.argmax(axis=1)

    metricas_dnn = evaluar_modelo(y_test, dnn_pred, dnn_prob, partidas, "dnn")
    print(f"  macro-F1: {metricas_dnn['macro_f1']}")
    print(f"  accuracy: {metricas_dnn['accuracy']}")
    print(f"  error_grave: {metricas_dnn['error_grave']}")

    # OOV
    print("\n=== Evaluación OOV ===")
    X_oov, y_oov_str, partidas_train = generar_features_oov()

    oov_idx = []
    for p in y_oov_str:
        matches = np.where(partidas_train == p)[0]
        oov_idx.append(matches[0] if len(matches) > 0 else -1)
    oov_idx = np.array(oov_idx)
    y_oov = oov_idx.copy()

    valid_mask = oov_idx >= 0
    print(f"  Items OOV válidos: {valid_mask.sum()} / {len(y_oov)}")

    logreg_oov_pred = logreg_model.predict(X_oov).astype(np.int64)
    logreg_oov_prob = logreg_model.predict_proba(X_oov)

    metricas_oov_logreg = evaluar_modelo(
        y_oov[valid_mask], logreg_oov_pred[valid_mask], logreg_oov_prob[valid_mask], partidas_train, "logreg_oov"
    )
    print(f"  logreg OOV macro-F1: {metricas_oov_logreg['macro_f1']}")
    print(f"  logreg OOV accuracy: {metricas_oov_logreg['accuracy']}")

    with torch.no_grad():
        logits_oov = dnn_model(torch.tensor(X_oov, dtype=torch.float32))
        dnn_oov_prob = torch.softmax(logits_oov, dim=1).numpy()
        dnn_oov_pred = dnn_oov_prob.argmax(axis=1)

    metricas_oov_dnn = evaluar_modelo(
        y_oov[valid_mask], dnn_oov_pred[valid_mask], dnn_oov_prob[valid_mask], partidas_train, "dnn_oov"
    )
    print(f"  dnn OOV macro-F1: {metricas_oov_dnn['macro_f1']}")
    print(f"  dnn OOV accuracy: {metricas_oov_dnn['accuracy']}")

    # Guardar métricas
    resultado = {
        "prueba": {"logreg": metricas_logreg, "dnn": metricas_dnn},
        "oov": {"logreg": metricas_oov_logreg, "dnn": metricas_oov_dnn},
        "hashes": {
            "embeddings": calcular_hash(EMBEDDINGS),
            "datos": calcular_hash(DATA_CLEAN / "DATOS.csv"),
            "oov": calcular_hash(RUTA_OOV),
        },
        "tamano_prueba": int(X_test.shape[0]),
        "tamano_oov": int(valid_mask.sum()),
    }

    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas en {METRICS}")

    # Review de errores
    review_dnn = review_errores(y_test, dnn_pred, partidas, labels_str_test, "DNN")
    review_logreg = review_errores(y_test, logreg_pred, partidas, labels_str_test, "Logreg")

    with open(ERROR_REVIEW, "w", encoding="utf-8") as f:
        f.write("# Review de errores\n\n")
        f.write(review_logreg)
        f.write("\n\n---\n\n")
        f.write(review_dnn)
    print(f"Review guardado en {ERROR_REVIEW}")

    # Ganador
    f1_logreg = metricas_logreg["macro_f1"]
    f1_dnn = metricas_dnn["macro_f1"]
    print(f"\n=== Resultado ===")
    print(f"Logreg macro-F1: {f1_logreg}")
    print(f"DNN macro-F1: {f1_dnn}")
    if f1_dnn > f1_logreg:
        print("Ganador: DNN")
    elif f1_logreg > f1_dnn:
        print("Ganador: Logreg")
    else:
        print("Empate → Logreg (más simple)")


if __name__ == "__main__":
    main()
