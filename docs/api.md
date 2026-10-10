Autor: Arando Espejo Alejandro Marcelo
Documentación de la API del clasificador de partidas de gasto.

## Iniciar el servidor

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

La API solo escucha en `127.0.0.1` (localhost). Sin autenticación.

## Endpoints

### GET /api/health

Estado del servicio.

**curl:**
```bash
curl http://127.0.0.1:8000/api/health
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health"
```

**Respuesta:**
```json
{"status": "ok", "modelos": ["dnn", "logreg"]}
```

---

### POST /api/predict

Clasifica una descripción de gasto.

**Parámetros:**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `item` | string | Sí | Descripción del gasto (máx. 500 caracteres) |

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"item": "papel bond t/oficio para impresion institucional"}'
```

**PowerShell:**
```powershell
$body = @{ item = "papel bond t/oficio para impresion institucional" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/predict" -Method Post -Body $body -ContentType "application/json"
```

**Respuesta (200):**
```json
{
  "partida_predicha": "32100",
  "nombre_predicho": "Papel de oficina y parecidos",
  "scores": [
    {"partida": "32100", "nombre": "Papel de oficina y parecidos", "score": 0.6521},
    {"partida": "32200", "nombre": "Cuadernos, carpetas y otros", "score": 0.1234},
    {"partida": "32300", "nombre": "Libros, folletos y otros", "score": 0.0567}
  ],
  "confianza": 0.6521,
  "elapsed_ms": 95.2
}
```

**Errores:**

| Código | Condición |
|--------|-----------|
| 422 | Campo `item` faltante |
| 422 | Texto vacío |
| 422 | Texto mayor a 500 caracteres |

---

### GET /api/metrics

Retorna las métricas de evaluación de los modelos (prueba y OOV).

**curl:**
```bash
curl http://127.0.0.1:8000/api/metrics
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/metrics"
```

**Respuesta:** archivo `artifacts/metrics.json` completo.

**Errores:**

| Código | Condición |
|--------|-----------|
| 404 | Archivo de métricas no encontrado |

---

## Notas

- Los scores provienen de softmax sobre la DNN; no son probabilidades calibradas.
- El modelo por defecto es la DNN (ganador por macro-F1).
- Top-3 partidas por predicción (configurable en `app/settings.py`).
