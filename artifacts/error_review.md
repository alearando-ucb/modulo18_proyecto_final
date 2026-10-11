# Review de errores

## Review de errores — Logreg

- Total errores: 35 / 202
- Errores graves: 8

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 22110 | 22120 | 3 |  |
| 43120 | 43700 | 2 |  |
| 39990 | 33100 | 2 |  |
| 25900 | 26990 | 2 |  |
| 26990 | 22210 | 2 |  |
| 34500 | 39300 | 2 |  |
| 25600 | 32300 | 1 | SI |
| 25600 | 22210 | 1 |  |
| 25210 | 39600 | 1 | SI |
| 43110 | 25500 | 1 |  |
| 34200 | 39100 | 1 |  |
| 39500 | 39800 | 1 |  |
| 39500 | 34600 | 1 |  |
| 39700 | 34500 | 1 |  |
| 39600 | 33100 | 1 |  |

### Ejemplos de errores graves

- `25600` → real `25600`, predicha `32300`
- `25210` → real `25210`, predicha `39600`
- `26990` → real `26990`, predicha `33100`
- `39800` → real `39800`, predicha `43120`
- `43700` → real `43700`, predicha `39700`
- `66210` → real `66210`, predicha `25400`
- `66210` → real `66210`, predicha `21100`
- `26610` → real `26610`, predicha `66210`

---

## Review de errores — DNN

- Total errores: 38 / 202
- Errores graves: 7

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 32100 | 32200 | 3 |  |
| 39990 | 33100 | 2 |  |
| 34800 | 34500 | 2 |  |
| 25900 | 26990 | 2 |  |
| 26990 | 22210 | 2 |  |
| 22120 | 22110 | 2 |  |
| 43120 | 34800 | 1 | SI |
| 22110 | 22210 | 1 |  |
| 25600 | 22110 | 1 |  |
| 25210 | 26990 | 1 |  |
| 33300 | 39400 | 1 |  |
| 34200 | 39100 | 1 |  |
| 24110 | 25900 | 1 |  |
| 39600 | 33100 | 1 |  |
| 26990 | 34800 | 1 | SI |

### Ejemplos de errores graves

- `43120` → real `43120`, predicha `34800`
- `26990` → real `26990`, predicha `34800`
- `39800` → real `39800`, predicha `43120`
- `43700` → real `43700`, predicha `39700`
- `66210` → real `66210`, predicha `25400`
- `66210` → real `66210`, predicha `21100`
- `26610` → real `26610`, predicha `66210`