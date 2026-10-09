Autor: Arando Espejo Alejandro Marcelo
Informe acumulado del clasificador presupuestario.

## Limpieza de la data

- Datos fuente: memoria de cálculo (Form 11 Presupuesto) de la FCBCB-CDL, gestiones 2020-2025, obtenidos en mayo de 2026. Información reservada.
- 1,292 filas reales, formato `;`, codificación `iso-8859-1`, 70 partidas únicas.
- Limpieza determinista aplicada en `01_prep_data.py`: minúsculas, remoción de ruido institucional (`cdl`, `fcbcb`, `revoluciones`, `jum`, `juaf`), normalización de espacios múltiples.
- 3 items con etiqueta conflictiva en los datos originales (mismo texto, partidas distintas): `papel japonés`, `mango de bisturí`, `silicona fría transparente`. Se resolvieron manualmente directamente en los datos de entrada antes del procesamiento.
- Umbral ≥6 registros por partida sobre el dataset combinado (real + sintético). El conteo se hace sobre el código de `PARTIDA`, no sobre el texto.
- Con solo datos reales, 23 partidas se habrían descartado. Con los sintéticos, solo se descarta 1 (66220).
- Dataset final: 1,450 filas, 70 partidas únicas.
- El grupo macro (primer dígito de la partida) se deriva únicamente para la métrica de gravedad, nunca como feature de entrada.

### Duplicados en el dataset

De las 1,450 filas finales, 352 son duplicados exactos (mismo texto limpio + misma partida). Esto no es un error del procesamiento: los datos originales de la memoria de cálculo contienen registros repetidos, lo cual es esperable en un presupuesto donde el mismo ítem de gasto se programa en múltiples unidades o gestiones. El experimento de referencia (Colab) también conserva estos duplicados, por lo que se mantiene el mismo comportamiento. Se generan IDs únicos en el manifiesto agregando un índice al hash para evitar colisiones.

## Datos sintéticos

- 159 registros sintéticos, versionados en `data/raw/DATOS SINTETICOS.csv`.
- Generados manualmente a partir del experimento de referencia (Colab, DNN versión 2).
- Objetivo: cubrir partidas con pocos registros reales para que superen el umbral ≥6 y no se descarten del entrenamiento.
- Partidas cubiertas: pasajes nacionales e internacionales, materiales de limpieza, contraste semántico (plástico vs eléctrico vs metal vs herramienta vs repuesto), lubricantes, cuero, muebles, gas, libros, agua, textiles, vajilla, prensa, seguros, laboratorio, energía eléctrica y trámites.
- Se combinan con los datos reales antes de aplicar el umbral.
