Autor: Arando Espejo Alejandro Marcelo
Documentación técnica de los datos del clasificador presupuestario.

## Origen de los datos

| Campo | Detalle |
|---|---|
| Fuente | Memoria de cálculo (Form 11 Presupuesto) de la FCBCB-CDL |
| Periodo | Gestiones 2020 al 2025 |
| Fecha de obtención | Mayo de 2026 |
| Licencia | Información reservada. No puede compartirse fuera de la institución |

## Estructura del archivo original

- **Archivo:** `data/raw/DATOS ENTRENAMIENTO.csv`
- **Formato:** separador `;`, codificación `iso-8859-1`
- **Columnas:** `ITEM` (descripción del gasto) y `PARTIDA` (código de 5 dígitos del Clasificador por Objeto del Gasto del MEF)
- **Registros:** 1,292 filas, 70 partidas únicas

## Conflicto partidas del grupo 20000 y 60000

Existia un conflicto entre los grupos macro 20000 y 60000, ya que ambos son pagos con las mismas categorías, la diferencia es la temporalidad, el grupo 60000 se usa para inscribir pagos de la gestión pasada.
En el dataset de origen no existe diferencia entre las descripciones de gastos de ambos grupos, por lo cual todo se clasificaba en el grupo 60000 ya que es el grupo mayoritario de los dos.
Como solución se modificó la descripción de los items del grupo 60000 agregando palabras que denotan temporalidad, se creo el archivo **DATA ENTRENAMIENTO V2.csv** que contiene estos cambios.

## Datos sintéticos

- **Archivo:** `data/raw/DATOS SINTETICOS.csv`
- **Registros:** 159 filas
- **Origen:** generados manualmente a partir del experimento de referencia (Colab, versión DNN 2)
- **Objetivo:** cubrir partidas con pocos registros reales para que superen el umbral mínimo y no se descarten
- **Partidas cubiertas:** pasajes (22110, 22120), materiales (34500, 34600, 34800, 39700, 39800), lubricantes (34110), cuero (34400), muebles (43110), gas (21500), libros (32300), agua (31120), textiles (33200), vajilla (39300), prensa (32500), seguros (22500), laboratorio (25120), energía eléctrica (21200), y trámites (85900)

## Limpieza de la data

- Minúsculas
- Eliminación de ruido institucional: `cdl`, `fcbcb`, `revoluciones`, `jum`, `juaf`
- Normalización de espacios múltiples

## Umbral mínimo

Se aplica umbral de **≥6 registros por partida** sobre el dataset combinado (real + sintético). El conteo se hace sobre el código de `PARTIDA`, no sobre el texto del ítem.

### Efecto del umbral

| Escenario | Partidas totales | Partidas descartadas |
|---|---|---|
| Solo datos reales | 70 | 23 |
| Real + sintéticos | 71 | 1 |

Los datos sintéticos rescataron 22 partidas que habrían sido descartadas.

### Partidas descartadas (solo datos reales, sin sintéticos)

| Partida | Registros |
|---|---|
| 11220 | 5 |
| 11400 | 5 |
| 11600 | 4 |
| 11700 | 5 |
| 13110 | 5 |
| 13120 | 5 |
| 13131 | 5 |
| 13200 | 5 |
| 21500 | 1 |
| 22500 | 2 |
| 25120 | 2 |
| 25700 | 4 |
| 26200 | 5 |
| 31120 | 2 |
| 32300 | 1 |
| 32500 | 2 |
| 33200 | 2 |
| 34110 | 1 |
| 34400 | 1 |
| 39300 | 2 |
| 43110 | 1 |
| 66100 | 5 |
| 66220 | 1 |

### Única partida descartada con sintéticos

| Partida | Registros | Nota |
|---|---|---|
| 66220 | 1 | Sin datos sintéticos de respaldo |

## Dataset final

- **Archivo:** `data/clean/DATOS.csv`
- **Registros:** 1,450 filas (1,292 reales + 159 sintéticos - 1 descartada)
- **Partidas únicas:** 70

## Manifest

- **Archivo:** `data/clean/manifest.json`
- **Campos por entrada:** `id` (hash corto), `group_id` (primer dígito de la partida), `split`, `label` (código de partida), `class_name` (texto limpio), `sha256` (hash completo del texto limpio)

## Reglas de fuga

Ningún derivado de `PARTIDA` se usa como feature de entrada al modelo. El código de partida solo se usa como etiqueta de salida y para agrupar en el split.
