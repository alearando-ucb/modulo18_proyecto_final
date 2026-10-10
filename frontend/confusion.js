// Autor: Arando Espejo Alejandro Marcelo
// Clasificador de Partidas de Gasto — AI Capstone Project
// confusion.js — Gráfico de matriz de confusión (heatmap en canvas).

let confusionData = null;

function renderizarConfusion(metrics, modelo, dataset, canvasId) {
  confusionData = metrics;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const data = metrics[dataset][modelo];
  const cm = data.matriz_confusion;
  const porClase = data.por_clase;

  const partidas = Object.keys(porClase);

  const conErrores = partidas.map((p, i) => {
    const support = cm[i].reduce((a, b) => a + b, 0);
    const correcto = cm[i][i];
    const fn = support - correcto;
    let fp = 0;
    for (let r = 0; r < cm.length; r++) fp += cm[r][i];
    fp -= correcto;
    return { p, errores: fn + fp };
  });

  conErrores.sort((a, b) => b.errores - a.errores);

  const top = conErrores.slice(0, 20).map(x => x.p);

  const indices = top.map(p => partidas.indexOf(p));

  const sub = indices.map(i => indices.map(j => cm[i] ? cm[i][j] || 0 : 0));

  dibujarHeatmap(canvas, sub, top);
}

function renderizarTablaErrores(metrics, modelo, dataset, tableId) {
  const cont = document.getElementById(tableId);
  if (!cont) return;

  const data = metrics[dataset][modelo];
  const cm = data.matriz_confusion;
  const porClase = data.por_clase;
  const partidas = Object.keys(porClase);

  const errores = [];
  for (let i = 0; i < partidas.length; i++) {
    const support = cm[i].reduce((a, b) => a + b, 0);
    const correcto = cm[i][i];
    const salen = support - correcto;
    let entran = 0;
    for (let r = 0; r < cm.length; r++) entran += cm[r][i];
    entran -= correcto;

    if (salen > 0 || entran > 0) {
      errores.push({ partida: partidas[i], support, correcto, salen, entran });
    }
  }

  errores.sort((a, b) => (b.salen + b.entran) - (a.salen + a.entran));

  if (errores.length === 0) {
    cont.innerHTML = "<p class='no-errors'>Sin errores detectados.</p>";
    return;
  }

  let html = `
    <h4 class="error-table-title">Partidas con errores (${errores.length})</h4>
    <table class="error-table">
      <thead>
        <tr>
          <th>Partida</th>
          <th>Ítems clasificados</th>
          <th>Aciertos</th>
          <th>Falsos negativos</th>
          <th>Falsos positivos</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const e of errores) {
    const nombre = partidasNombres[e.partida] || "";
    html += `
        <tr>
          <td title="${nombre}">${e.partida}</td>
          <td>${e.support}</td>
          <td>${e.correcto}</td>
          <td class="${e.salen > 0 ? "cell-error" : ""}">${e.salen}</td>
          <td class="${e.entran > 0 ? "cell-warn" : ""}">${e.entran}</td>
        </tr>
    `;
  }

  html += `
      </tbody>
    </table>
  `;

  cont.innerHTML = html;
}

function dibujarHeatmap(canvas, data, labels) {
  const n = labels.length;
  const cellSize = Math.min(28, Math.floor(500 / n));
  const padding = 60;
  const width = padding + n * cellSize + 20;
  const height = padding + n * cellSize + 20;

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  const maxVal = Math.max(...data.flat());

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = data[i][j];
      const intensity = maxVal > 0 ? val / maxVal : 0;

      const r = Math.round(255 - intensity * 200);
      const g = Math.round(255 - intensity * 200);
      const b = Math.round(255 - intensity * 50);

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(
        padding + j * cellSize,
        padding + i * cellSize,
        cellSize,
        cellSize
      );

      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(
        padding + j * cellSize,
        padding + i * cellSize,
        cellSize,
        cellSize
      );

      if (val > 0) {
        ctx.fillStyle = intensity > 0.5 ? "white" : "#333";
        ctx.font = `${Math.min(11, cellSize * 0.4)}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          val,
          padding + j * cellSize + cellSize / 2,
          padding + i * cellSize + cellSize / 2
        );
      }
    }
  }

  ctx.fillStyle = "#333";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let j = 0; j < n; j++) {
    ctx.save();
    ctx.translate(
      padding + j * cellSize + cellSize / 2,
      padding - 5
    );
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(labels[j], 0, 0);
    ctx.restore();
  }

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    ctx.fillText(
      labels[i],
      padding - 5,
      padding + i * cellSize + cellSize / 2
    );
  }

  ctx.fillStyle = "#1a237e";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("Real ↓ / Predicha →", 5, 5);
}
