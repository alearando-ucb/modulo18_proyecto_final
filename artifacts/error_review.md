# Review de errores

## Review de errores — Logreg

- Total errores: 50 / 275
- Errores graves: 15

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 21200 | 39700 | 3 | SI |
| 21300 | 34200 | 3 | SI |
| 34500 | 39700 | 3 |  |
| 39500 | 34500 | 3 |  |
| 22110 | 22120 | 3 |  |
| 39500 | 34800 | 2 |  |
| 39500 | 32200 | 2 |  |
| 26990 | 25900 | 1 |  |
| 34800 | 39990 | 1 |  |
| 22300 | 22110 | 1 |  |
| 25210 | 39600 | 1 | SI |
| 39990 | 32300 | 1 |  |
| 25600 | 26990 | 1 |  |
| 34500 | 31300 | 1 |  |
| 21100 | 66210 | 1 | SI |

### Ejemplos de errores graves

- `21200` → real `21200`, predicha `39700`
- `21300` → real `21300`, predicha `34200`
- `25210` → real `25210`, predicha `39600`
- `21200` → real `21200`, predicha `39700`
- `21300` → real `21300`, predicha `34200`
- `21200` → real `21200`, predicha `39700`
- `21300` → real `21300`, predicha `34200`
- `21100` → real `21100`, predicha `66210`
- `39500` → real `39500`, predicha `25500`
- `39990` → real `39990`, predicha `43700`

---

## Review de errores — DNN

- Total errores: 32 / 275
- Errores graves: 7

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 21300 | 34200 | 3 | SI |
| 32200 | 32100 | 2 |  |
| 34500 | 34600 | 2 |  |
| 39500 | 32200 | 2 |  |
| 39600 | 22210 | 1 | SI |
| 34800 | 33100 | 1 |  |
| 25210 | 26990 | 1 |  |
| 34800 | 34500 | 1 |  |
| 39100 | 33100 | 1 |  |
| 39500 | 34200 | 1 |  |
| 39500 | 34800 | 1 |  |
| 39500 | 34500 | 1 |  |
| 39500 | 39990 | 1 |  |
| 39700 | 34500 | 1 |  |
| 39400 | 34500 | 1 |  |

### Ejemplos de errores graves

- `39600` → real `39600`, predicha `22210`
- `21300` → real `21300`, predicha `34200`
- `21300` → real `21300`, predicha `34200`
- `21300` → real `21300`, predicha `34200`
- `43700` → real `43700`, predicha `34800`
- `43700` → real `43700`, predicha `34600`
- `43110` → real `43110`, predicha `34600`