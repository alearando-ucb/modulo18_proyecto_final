# Review de errores

## Review de errores — Logreg

- Total errores: 48 / 232
- Errores graves: 11

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 66210 | 66400 | 5 |  |
| 39500 | 34800 | 4 |  |
| 32200 | 32100 | 2 |  |
| 39500 | 34500 | 2 |  |
| 26990 | 25600 | 1 |  |
| 39600 | 26990 | 1 | SI |
| 25600 | 22210 | 1 |  |
| 32200 | 25500 | 1 | SI |
| 39500 | 43110 | 1 | SI |
| 34800 | 31300 | 1 |  |
| 34800 | 39990 | 1 |  |
| 25210 | 26990 | 1 |  |
| 24120 | 43120 | 1 |  |
| 26990 | 33100 | 1 | SI |
| 39700 | 26990 | 1 | SI |

### Ejemplos de errores graves

- `39600` → real `39600`, predicha `26990`
- `32200` → real `32200`, predicha `25500`
- `39500` → real `39500`, predicha `43110`
- `26990` → real `26990`, predicha `33100`
- `39700` → real `39700`, predicha `26990`
- `25500` → real `25500`, predicha `31300`
- `43700` → real `43700`, predicha `39700`
- `24300` → real `24300`, predicha `39700`
- `25900` → real `25900`, predicha `39600`
- `39800` → real `39800`, predicha `43700`

---

## Review de errores — DNN

- Total errores: 29 / 232
- Errores graves: 6

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 66210 | 66400 | 5 |  |
| 32200 | 39500 | 3 |  |
| 39600 | 26990 | 1 | SI |
| 34800 | 33300 | 1 |  |
| 34800 | 39500 | 1 |  |
| 34800 | 34110 | 1 |  |
| 25210 | 26990 | 1 |  |
| 24120 | 39800 | 1 | SI |
| 26990 | 39500 | 1 | SI |
| 22110 | 22600 | 1 |  |
| 39500 | 34800 | 1 |  |
| 39400 | 34800 | 1 |  |
| 24300 | 39700 | 1 | SI |
| 25500 | 39500 | 1 | SI |
| 43700 | 39500 | 1 | SI |

### Ejemplos de errores graves

- `39600` → real `39600`, predicha `26990`
- `24120` → real `24120`, predicha `39800`
- `26990` → real `26990`, predicha `39500`
- `24300` → real `24300`, predicha `39700`
- `25500` → real `25500`, predicha `39500`
- `43700` → real `43700`, predicha `39500`