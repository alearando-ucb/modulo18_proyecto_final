# Clasificador de Partidas de Gasto

Proyecto final de la materia **ARTIFICIAL INTELLIGENCE CAPSTONE PROJECT**.

- Maestrante: **Arando Espejo Alejandro Marcelo**

## Descripción

Sistema de clasificación automática de partidas de gasto. Dada la descripción de un ítem de gasto, el modelo asigna la partida correspondiente del Clasificador por Objeto del Gasto del Ministerio de Economía y Finanzas Públicas de Bolivia.

El clasificador oficial agrupa los gastos públicos por su naturaleza u objeto (por ejemplo, materiales de limpieza, pasajes y viáticos, servicios personales). La asignación de cada ítem a su partida se realiza hoy de forma manual y requiere conocer las definiciones del clasificador. Este proyecto propone un asistente automático para esa tarea.

## Enfoque

- Datos: ítems de gasto reales de la FCBCB - CDL, con su partida asignada.
- Referencia normativa: Clasificador por Objeto del Gasto (Ministerio de Economía y   Finanzas Públicas), cuyas descripciones oficiales se utilizan como conocimiento de apoyo.
- Modelos: extracción de características con un modelo de lenguaje multilingüe congelado y comparación entre una red neuronal y una línea base de regresión logística sobre las mismos embeddings.
