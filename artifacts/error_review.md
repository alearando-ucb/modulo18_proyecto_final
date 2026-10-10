# Review de errores

## Review de errores — Logreg

- Total errores: 55 / 295
- Errores graves: 12

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 66210 | 66400 | 5 |  |
| 26990 | 25500 | 4 |  |
| 25900 | 26990 | 2 |  |
| 34500 | 39700 | 2 |  |
| 34500 | 39400 | 2 |  |
| 25120 | 13110 | 2 |  |
| 39500 | 34500 | 2 |  |
| 39700 | 34500 | 2 |  |
| 34800 | 39990 | 1 |  |
| 25600 | 39600 | 1 | SI |
| 34500 | 34200 | 1 |  |
| 22110 | 22600 | 1 |  |
| 34200 | 34800 | 1 |  |
| 34800 | 25500 | 1 | SI |
| 39500 | 43120 | 1 | SI |

### Ejemplos de errores graves

- `25600` → real `25600`, predicha `39600`
- `34800` → real `34800`, predicha `25500`
- `39500` → real `39500`, predicha `43120`
- `39800` → real `39800`, predicha `43120`
- `21100` → real `21100`, predicha `66210`
- `39500` → real `39500`, predicha `43110`
- `39990` → real `39990`, predicha `43700`
- `34500` → real `34500`, predicha `43700`
- `43700` → real `43700`, predicha `39700`
- `66210` → real `66210`, predicha `24120`

---

## Review de errores — DNN

- Total errores: 58 / 295
- Errores graves: 19

### Top confusiones

| Real | Predicha | Frecuencia | Grave |
|------|----------|------------|-------|
| 66100 | 11700 | 5 | SI |
| 66210 | 66400 | 5 |  |
| 26990 | 25600 | 5 |  |
| 21600 | 39700 | 4 | SI |
| 39500 | 34500 | 4 |  |
| 25900 | 26990 | 2 |  |
| 25120 | 13110 | 2 |  |
| 39600 | 26990 | 1 | SI |
| 34800 | 39100 | 1 |  |
| 25600 | 39600 | 1 | SI |
| 22110 | 22600 | 1 |  |
| 34200 | 34800 | 1 |  |
| 34800 | 39500 | 1 |  |
| 39500 | 33100 | 1 |  |
| 39990 | 34800 | 1 |  |

### Ejemplos de errores graves

- `39600` → real `39600`, predicha `26990`
- `25600` → real `25600`, predicha `39600`
- `66100` → real `66100`, predicha `11700`
- `21600` → real `21600`, predicha `39700`
- `66100` → real `66100`, predicha `11700`
- `21600` → real `21600`, predicha `39700`
- `66100` → real `66100`, predicha `11700`
- `39990` → real `39990`, predicha `43700`
- `34500` → real `34500`, predicha `43700`
- `21600` → real `21600`, predicha `39700`