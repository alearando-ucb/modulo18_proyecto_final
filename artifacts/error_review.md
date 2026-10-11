# Review de errores

## Review de errores — Logreg

- Total errores: 36 / 231
- Errores graves: 5

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 24110 | 24120 | 3 |  |
| 43120 | 43700 | 2 |  |
| 22210 | 22110 | 2 |  |
| 34600 | 34800 | 2 |  |
| 25900 | 26990 | 2 |  |
| 39500 | 34800 | 2 |  |
| 39800 | 34800 | 2 |  |
| 39400 | 34800 | 2 |  |
| 39500 | 34500 | 2 |  |
| 39990 | 33100 | 1 |  |
| 25210 | 39600 | 1 | SI |
| 34600 | 33100 | 1 |  |
| 39500 | 39400 | 1 |  |
| 39500 | 39800 | 1 |  |
| 39700 | 34800 | 1 |  |

### Ejemplos de errores graves

- `25210` → real `25210`, predicha `39600`
- `26990` → real `26990`, predicha `33100`
- `43500` → real `43500`, predicha `39700`
- `43700` → real `43700`, predicha `34800`
- `43700` → real `43700`, predicha `39700`

---

## Review de errores — DNN

- Total errores: 62 / 231
- Errores graves: 10

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 39500 | 39800 | 8 |  |
| 26610 | 22600 | 6 |  |
| 24120 | 24110 | 4 |  |
| 39500 | 34500 | 4 |  |
| 39500 | 34800 | 3 |  |
| 22210 | 22110 | 2 |  |
| 25900 | 26990 | 2 |  |
| 39400 | 34800 | 2 |  |
| 34500 | 39300 | 2 |  |
| 39500 | 43500 | 2 | SI |
| 43120 | 43700 | 1 |  |
| 39990 | 34200 | 1 |  |
| 25210 | 26990 | 1 |  |
| 24120 | 39800 | 1 | SI |
| 66210 | 25900 | 1 | SI |

### Ejemplos de errores graves

- `24120` → real `24120`, predicha `39800`
- `66210` → real `66210`, predicha `25900`
- `34600` → real `34600`, predicha `43110`
- `32200` → real `32200`, predicha `43110`
- `39500` → real `39500`, predicha `43500`
- `43500` → real `43500`, predicha `39700`
- `43700` → real `43700`, predicha `34800`
- `43700` → real `43700`, predicha `39700`
- `39500` → real `39500`, predicha `43500`
- `43120` → real `43120`, predicha `39800`