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
