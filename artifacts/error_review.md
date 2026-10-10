# Review de errores

## Review de errores — Logreg

- Total errores: 54 / 229
- Errores graves: 18

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 26990 | 32300 | 4 | SI |
| 39500 | 34500 | 4 |  |
| 21300 | 34200 | 3 | SI |
| 39990 | 33100 | 2 |  |
| 34800 | 34600 | 2 |  |
| 39500 | 32200 | 2 |  |
| 34500 | 34800 | 2 |  |
| 32200 | 32100 | 2 |  |
| 34200 | 39500 | 2 |  |
| 32200 | 34800 | 1 |  |
| 34800 | 31300 | 1 |  |
| 24110 | 31300 | 1 | SI |
| 39700 | 39600 | 1 |  |
| 34200 | 34800 | 1 |  |
| 39700 | 21200 | 1 | SI |

### Ejemplos de errores graves

- `21300` → real `21300`, predicha `34200`
- `24110` → real `24110`, predicha `31300`
- `21300` → real `21300`, predicha `34200`
- `39700` → real `39700`, predicha `21200`
- `39800` → real `39800`, predicha `43120`
- `21300` → real `21300`, predicha `34200`
- `39500` → real `39500`, predicha `43110`
- `39800` → real `39800`, predicha `43700`
- `34500` → real `34500`, predicha `43700`
- `26990` → real `26990`, predicha `32300`

---

## Review de errores — DNN

- Total errores: 39 / 229
- Errores graves: 7

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 26990 | 25600 | 4 |  |
| 21300 | 31120 | 3 | SI |
| 24110 | 26990 | 2 |  |
| 34500 | 39990 | 2 |  |
| 34800 | 34600 | 2 |  |
| 34500 | 34800 | 2 |  |
| 34200 | 39500 | 2 |  |
| 39990 | 33100 | 1 |  |
| 32200 | 32100 | 1 |  |
| 34800 | 33300 | 1 |  |
| 34200 | 34800 | 1 |  |
| 39990 | 34200 | 1 |  |
| 34200 | 31120 | 1 |  |
| 39100 | 33200 | 1 |  |
| 39700 | 34500 | 1 |  |

### Ejemplos de errores graves

- `21300` → real `21300`, predicha `31120`
- `21300` → real `21300`, predicha `31120`
- `21300` → real `21300`, predicha `31120`
- `34500` → real `34500`, predicha `43700`
- `43500` → real `43500`, predicha `34600`
- `43700` → real `43700`, predicha `34800`
- `43110` → real `43110`, predicha `39500`