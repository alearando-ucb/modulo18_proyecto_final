// Autor: Arando Espejo Alejandro Marcelo
// Clasificador de Partidas de Gasto — AI Capstone Project
// app.js — Lógica del frontend: clasificación, ejemplos y métricas.

const EJEMPLOS = [
  {
    item: "pasaje aereo tramo sucre buenos aires para evento internacional",
    partida: "22120",
    nombre: "Pasajes aéros internacionales",
    frontera: "22110 vs 22120 (nacional vs internacional)"
  },
  {
    item: "pasajes aereos sucre-cochabamba-sucre",
    partida: "22110",
    nombre: "Pasajes aéros nacionales",
    frontera: "22110 vs 22120 (nacional vs internacional)"
  },
  {
    item: "viaticos cobija oruro",
    partida: "22210",
    nombre: "Viáticos nacionales",
    frontera: "22110 vs 22210 (pasaje vs viático)"
  },
  {
    item: "servicio de courier mensual",
    partida: "21100",
    nombre: "Servicio de courier",
    frontera: "21100 vs 26990 (servicio básico vs otros)"
  },
  {
    item: "papel bond t/oficio para impresion institucional",
    partida: "32100",
    nombre: "Papel de oficina",
    frontera: "32100 vs 32200 (papel vs cuaderno)"
  },
  {
    item: "foco led 15 watts luz blanca",
    partida: "39700",
    nombre: "Útiles y materiales eléctricos",
    frontera: "39700 vs 39800 (eléctrico vs repuesto)"
  },
  {
    item: "archivador metalico de cuatro gavetas",
    partida: "43110",
    nombre: "Equipo de oficina y muebles",
    frontera: "43110 vs 34600 (mueble vs producto metálico)"
  },
  {
    item: "monitor adicional para estacion de trabajo",
    partida: "43500",
    nombre: "Equipo de comunicación",
    frontera: "43500 vs 39800 (equipo vs repuesto)"
  }
];

// Navegación entre secciones
document.querySelectorAll("nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.section).classList.add("active");

    if (btn.dataset.section === "evaluacion") {
      cargarMetricas();
    }
  });
});

// Botón clasificar
document.getElementById("btn-predict").addEventListener("click", () => {
  const texto = document.getElementById("item-input").value.trim();
  if (!texto) {
    mostrarError("predict-result", "Escriba un texto para clasificar.");
    return;
  }
  clasificar(texto, "predict-result");
});

async function clasificar(texto, contenedorId) {
  const contenedor = document.getElementById(contenedorId);
  contenedor.innerHTML = "<p>Clasificando...</p>";

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item: texto })
    });

    if (!resp.ok) {
      const err = await resp.json();
      mostrarError(contenedorId, err.detail || "Error en la predicción");
      return;
    }

    const data = await resp.json();
    mostrarResultado(data, contenedorId);
  } catch (e) {
    mostrarError(contenedorId, "No se pudo conectar con la API.");
  }
}

function mostrarResultado(data, contenedorId) {
  const contenedor = document.getElementById(contenedorId);
  let html = `
    <div class="result">
      <div class="prediction">
        <strong>${data.partida_predicha}</strong> — ${data.nombre_predicho}
      </div>
      <p>Confianza: ${(data.confianza * 100).toFixed(1)}% · ${data.elapsed_ms} ms</p>
      <h3 style="margin-top:15px;">Top-${data.scores.length}</h3>
  `;

  data.scores.forEach(s => {
    const pct = (s.score * 100).toFixed(1);
    html += `
      <div class="score-bar">
        <span class="score-label">${s.partida} — ${s.nombre}</span>
        <span class="score-value">${pct}%</span>
        <div class="bar-container">
          <div class="bar-fill" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  });

  html += "</div>";
  contenedor.innerHTML = html;
}

function mostrarError(contenedorId, mensaje) {
  document.getElementById(contenedorId).innerHTML = `<div class="error">${mensaje}</div>`;
}

// Ejemplos
function renderizarEjemplos() {
  const lista = document.getElementById("examples-list");
  lista.innerHTML = "";
  EJEMPLOS.forEach((ej, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="item-text">${ej.item}</div>
      <div class="item-expected">Real: ${ej.partida} — ${ej.nombre} · Frontera: ${ej.frontera}</div>
    `;
    li.addEventListener("click", () => clasificar(ej.item, "example-result"));
    lista.appendChild(li);
  });
}

// Métricas
let metricsCache = null;
let partidasNombres = {};

fetch("/api/partidas")
  .then(r => r.json())
  .then(d => { partidasNombres = d; })
  .catch(() => {});

async function cargarMetricas() {
  try {
    const resp = await fetch("/api/metrics");
    const data = await resp.json();
    metricsCache = data;
    renderizarMetricas(data);
    actualizarGraficos();
  } catch (e) {
    document.getElementById("metrics-content").innerHTML = "<p>Error al cargar métricas.</p>";
  }
}

function actualizarGraficos() {
  if (!metricsCache) return;
  const m1 = document.getElementById("select-modelo-1").value;
  const d1 = document.getElementById("select-dataset-1").value;
  const m2 = document.getElementById("select-modelo-2").value;
  const d2 = document.getElementById("select-dataset-2").value;
  renderizarConfusion(metricsCache, m1, d1, "confusion-chart-1");
  renderizarTablaErrores(metricsCache, m1, d1, "error-table-1");
  renderizarConfusion(metricsCache, m2, d2, "confusion-chart-2");
  renderizarTablaErrores(metricsCache, m2, d2, "error-table-2");
}

["select-modelo-1", "select-dataset-1", "select-modelo-2", "select-dataset-2"].forEach(id => {
  document.getElementById(id).addEventListener("change", actualizarGraficos);
});

function renderizarMetricas(data) {
  const p = data.prueba;
  const o = data.oov;

  const html = `
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">Macro-F1 (prueba)</div>
        <div class="value">${p.dnn.macro_f1}</div>
        <div class="model">DNN</div>
      </div>
      <div class="metric-card">
        <div class="label">Macro-F1 (prueba)</div>
        <div class="value">${p.logreg.macro_f1}</div>
        <div class="model">Logreg</div>
      </div>
      <div class="metric-card">
        <div class="label">Error grave (prueba)</div>
        <div class="value">${(p.dnn.error_grave * 100).toFixed(2)}%</div>
        <div class="model">DNN</div>
      </div>
      <div class="metric-card">
        <div class="label">Error grave (prueba)</div>
        <div class="value">${(p.logreg.error_grave * 100).toFixed(2)}%</div>
        <div class="model">Logreg</div>
      </div>
    </div>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">Macro-F1 (OOV)</div>
        <div class="value">${o.dnn.macro_f1}</div>
        <div class="model">DNN</div>
      </div>
      <div class="metric-card">
        <div class="label">Macro-F1 (OOV)</div>
        <div class="value">${o.logreg.macro_f1}</div>
        <div class="model">Logreg</div>
      </div>
      <div class="metric-card">
        <div class="label">Error grave (OOV)</div>
        <div class="value">${(o.dnn.error_grave * 100).toFixed(2)}%</div>
        <div class="model">DNN</div>
      </div>
      <div class="metric-card">
        <div class="label">Error grave (OOV)</div>
        <div class="value">${(o.logreg.error_grave * 100).toFixed(2)}%</div>
        <div class="model">Logreg</div>
      </div>
    </div>
    <p style="margin-top:10px; color:#666;">
      Prueba: ${data.tamano_prueba} ítems · OOV: ${data.tamano_oov} ítems
    </p>
  `;
  document.getElementById("metrics-content").innerHTML = html;
}

renderizarEjemplos();
