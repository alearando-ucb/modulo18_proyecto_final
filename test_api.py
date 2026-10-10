import urllib.request, json, time

data = json.dumps({"item": "papel bond t/oficio"}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/predict",
    data=data,
    headers={"Content-Type": "application/json"},
)
start = time.time()
resp = urllib.request.urlopen(req)
elapsed = time.time() - start
result = json.loads(resp.read())
print("Tiempo: %.2fs" % elapsed)
print("Partida: %s - %s" % (result["partida_predicha"], result["nombre_predicho"]))
print("Confianza:", result["confianza"])
print("Top-3:")
for s in result["scores"]:
    print("  %s: %s (%s)" % (s["partida"], s["nombre"], s["score"]))
