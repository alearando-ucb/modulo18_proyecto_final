# Clasificador de Partidas de Gasto

Proyecto final de la materia **ARTIFICIAL INTELLIGENCE CAPSTONE PROJECT**.

- Maestrante: **Arando Espejo Alejandro Marcelo**

## Descripción

Sistema de clasificación automática de partidas de gasto. Dada la descripción de un ítem de gasto, el modelo asigna la partida correspondiente del Clasificador por Objeto del Gasto del Ministerio de Economía y Finanzas Públicas de Bolivia.

El clasificador oficial agrupa los gastos públicos por su naturaleza u objeto (por ejemplo, materiales de limpieza, pasajes y viáticos, servicios personales). La asignación de cada ítem a su partida se realiza hoy de forma manual y requiere conocer las definiciones del clasificador. Este proyecto propone un asistente automático para esa tarea.

## Problema

Asignar automáticamente la partida de gasto a cada descripción de ítem (tóner, pasajes, y en general cualquier concepto de gasto), evitando confusiones graves entre grupos, como confundir materiales (3000) con activos (4000).

La solución actual, la asignación manual por expertos del área, es correcta pero depende de la disponibilidad de las personas con experiencia.

## Unidad de predicción

La descripción de gasto con la partida asociada por el experto. Cada fila del conjunto de datos corresponde a un ítem de gasto.

## Enfoque

- Datos: ítems de gasto reales de la FCBCB - CDL, con su partida asignada.
- Referencia normativa: Clasificador por Objeto del Gasto (Ministerio de Economía y Finanzas Públicas), cuyas descripciones oficiales se utilizan como conocimiento de apoyo.
- Modelos: extracción de características con un modelo de lenguaje multilingüe congelado y comparación entre una red neuronal y una línea base de regresión logística sobre los mismos embeddings.

## Justificación de la solución propuesta
Con pocos ejemplos por partida, un modelo basado en vocabulario (TF-IDF) no generaliza, esto quedó demostrado en la primera versión del clasificador basado en tf-idf + machine learning; al contrario los embeddings preentrenados aportan la semántica de las descripciones, a la par que permiten trabajar con valores fuera del vocabulario.

## Métricas

- Macro-F1 de la partida: se reporta siempre.
- Tasa de errores graves predefinidos entre grupos sobre el total evaluado: los pares de confusión graves, la gravedad y el denominador se definen y congelan antes de entrenar.
- La métrica principal se define al congelar los errores graves.

## Instalación

Requisitos: Python 3.10 o superior. Se recomienda una máquina con GPU NVIDIA para el entrenamiento de la red neuronal, ya que el proceso en CPU puede tardar varias horas. La inferencia de la API funciona sin GPU.

Para crear el entorno virtual e instalar las dependencias:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Las dependencias principales son pandas, numpy, scikit-learn, torch, sentence-transformers, fastapi y uvicorn.

## Estructura del proyecto

```
├── app/
│   ├── 01_prep_data.py        # Limpieza y preparación de datos
│   ├── 02_features.py         # Generación de embeddings y features
│   ├── 03_train.py            # Entrenamiento de la red neuronal
│   ├── 04_evaluate.py         # Evaluación en prueba y OOV
│   ├── logreg.py              # Regresión logística (línea base)
│   ├── dnn.py                 # Arquitectura de la red neuronal
│   ├── predict.py             # Lógica de inferencia
│   ├── main.py                # Servidor FastAPI
│   └── settings.py            # Constantes y rutas
├── data/
│   ├── raw/                   # Datos de entrada
│   ├── clean/                 # Datos limpios y nombres de partidas
│   └── oov/                   # Conjunto fuera de vocabulario
├── artifacts/                 # Modelos y métricas entrenadas
├── frontend/                  # Interfaz web
├── docs/                      # Documentación de la API y los datos
├── run.py                     # Orquestador del pipeline
└── requirements.txt
```

## Ejecución

Para regenerar todo el pipeline (datos, embeddings, modelos y métricas) y levantar la API con un solo comando:

```powershell
python run.py --rebuild
```

El orquestador ejecuta los cinco pasos del pipeline en orden y al finalizar inicia el servidor. Opciones:

- `--rebuild` fuerza todos los pasos; sin él, solo se ejecutan los pasos cuyos artefactos faltan o quedaron desactualizados (compara los hashes de `DATOS.csv`, de los embeddings y del OOV contra los registrados en `metrics.json`).
- `--dry-run` muestra el plan de pasos sin ejecutar nada.
- `--no-serve` regenera y termina sin iniciar la API.
- `--port` cambia el puerto de la API (por defecto 8000).

Los pasos individuales, en orden, son:

```powershell
python -m app.01_prep_data    # Preparar datos limpios
python -m app.02_features     # Generar embeddings (recomendado con GPU)
python -m app.logreg          # Entrenar línea base
python -m app.03_train        # Entrenar red neuronal (recomendado con GPU)
python -m app.04_evaluate     # Evaluar ambos modelos
```

Si la máquina no tiene GPU, se recomienda correr `python run.py --rebuild` en la máquina con GPU y subir los artefactos resultantes de `artifacts/` al repositorio.

Para iniciar solo la API y el frontend, sin regenerar artefactos:

```powershell
.\venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```

Luego abrir http://127.0.0.1:8000/ en el navegador. La interfaz ofrece un clasificador de ítems, ejemplos de prueba, un panel de evaluación con matrices de confusión comparables entre modelos y conjuntos, y un botón de corrección en cada predicción que registra la partida correcta para el reentrenamiento.

## Resultados

Los modelos se evalúan en dos conjuntos: prueba (284 ítems con distribución similar al entrenamiento) y OOV (140 ítems nunca vistos, 2 por partida).

| Métrica | DNN | Regresión logística |
|---------|-----|---------------------|
| Macro-F1 (prueba) | 0.8121 | 0.7879 |
| Macro-F1 (OOV) | 0.7571 | 0.7463 |
| Error grave (prueba) | 1.76% | 3.52% |
| Error grave (OOV) | 5.00% | 6.43% |

La DNN es el modelo ganador según el criterio congelado (mayor Macro-F1) y presenta menor tasa de error grave en ambos conjuntos.

## API

La API expone cinco endpoints:

- `GET /api/health` — estado del servicio.
- `POST /api/predict` — clasifica un ítem de gasto. Acepta `{"item": "texto del gasto"}` y retorna la partida predicha, su nombre y el top-3 de puntajes.
- `GET /api/metrics` — métricas de evaluación de ambos modelos.
- `GET /api/partidas` — catálogo de partidas (código y nombre) usado por el selector de correcciones.
- `POST /api/feedback` — registra una corrección de predicción. Acepta `{"item", "partida_predicha", "partida_correcta", "usuario"}` y la agrega a `DATOS CORRECCIONES.csv` para el siguiente reentrenamiento.

Ejemplo de predicción:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/predict" -Method Post -ContentType "application/json" -Body '{"item": "papel bond t/oficio"}'
```

La documentación interactiva de la API está disponible en http://127.0.0.1:8000/docs. Para más detalles consultar `docs/api.md`.
