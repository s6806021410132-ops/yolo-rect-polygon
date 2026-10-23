// Client-side YOLO11 (detect + segment) inference with onnxruntime-web.
const NAMES = ["person", "car", "motorcycle", "cat", "dog"];
const COLORS = ["#e6194b", "#3cb44b", "#f58231", "#4363d8", "#911eb4"];
const SIZE = 640;
const MODELS = { seg: "models/seg.onnx", det: "models/det.onnx" };
const SAMPLES = ["samples/1.jpg", "samples/2.jpg", "samples/3.jpg", "samples/4.jpg", "samples/5.jpg", "samples/6.jpg"];

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const sessions = {};
let mode = "seg", currentImg = null, lastRaw = null;

function setStatus(t) { $("status").textContent = t; }

async function getSession(m) {
  if (!sessions[m]) {
    setStatus(`กำลังโหลดโมเดล ${m}…`);
    sessions[m] = await ort.InferenceSession.create(MODELS[m], { executionProviders: ["wasm"] });
  }
  return sessions[m];
}

// Letterbox image into a 640x640 RGB float tensor (CHW, 0..1)
function preprocess(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const r = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);
  const c = new OffscreenCanvas(SIZE, SIZE), g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)";
  g.fillRect(0, 0, SIZE, SIZE);
  g.drawImage(img, px, py, nw, nh);
  const d = g.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = d[i * 4] / 255;
    f[n + i] = d[i * 4 + 1] / 255;
    f[2 * n + i] = d[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]), r, px, py, nw, nh };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
}

// output0: [1, 4 + nc (+32 mask coefs), 8400]
function decode(out0, conf, iouT) {
  const [, ch, na] = out0.dims, d = out0.data, nc = NAMES.length;
  const cands = [];
  for (let i = 0; i < na; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) {
      const s = d[(4 + c) * na + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < conf) continue;
    const cx = d[i], cy = d[na + i], w = d[2 * na + i], h = d[3 * na + i];
    const coef = [];
    for (let k = 4 + nc; k < ch; k++) coef.push(d[k * na + i]);
    cands.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls, coef });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cands) {
    if (keep.every((k) => k.cls !== c.cls || iou(k, c) < iouT)) keep.push(c);
    if (keep.length >= 100) break;
  }
  return keep;
}

// mask = sigmoid(coef · protos), cropped to the box, as a 160x160 RGBA canvas
function buildMask(det, protos) {
  const [, nm, mh, mw] = protos.dims, p = protos.data, sx = mw / SIZE, sy = mh / SIZE;
  const c = new OffscreenCanvas(mw, mh), g = c.getContext("2d");
  const im = g.createImageData(mw, mh);
  const col = hexRgb(COLORS[det.cls]);
  const bx1 = det.x1 * sx, bx2 = det.x2 * sx, by1 = det.y1 * sy, by2 = det.y2 * sy;
  for (let y = Math.max(0, Math.floor(by1)); y < Math.min(mh, Math.ceil(by2)); y++) {
    for (let x = Math.max(0, Math.floor(bx1)); x < Math.min(mw, Math.ceil(bx2)); x++) {
      const idx = y * mw + x;
      let v = 0;
      for (let k = 0; k < nm; k++) v += det.coef[k] * p[k * mh * mw + idx];
      if (v > 0) { // sigmoid(v) > 0.5
        im.data[idx * 4] = col[0]; im.data[idx * 4 + 1] = col[1]; im.data[idx * 4 + 2] = col[2]; im.data[idx * 4 + 3] = 255;
      }
    }
  }
  g.putImageData(im, 0, 0);
  return c;
}

function hexRgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }

async function run() {
  if (!currentImg) return;
  const sess = await getSession(mode);
  setStatus("กำลังทาย…");
  const pre = preprocess(currentImg);
  const t0 = performance.now();
  const out = await sess.run({ [sess.inputNames[0]]: pre.tensor });
  const ms = performance.now() - t0;
  lastRaw = { out, pre, ms, mode, names: sess.outputNames };
  render();
}

function render() {
  if (!lastRaw) return;
  const { out, pre, ms, names } = lastRaw;
  const conf = +$("conf").value, iouT = +$("iou").value;
  const out0 = out[names[0]], protos = names.length > 1 ? out[names[1]] : null;
  const dets = decode(out0, conf, iouT);
  const img = currentImg, W = img.naturalWidth, H = img.naturalHeight;
  cv.width = W; cv.height = H; cv.hidden = false; $("placeholder").hidden = true;
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(W, H) / 300));
  const fs = Math.max(12, Math.round(Math.max(W, H) / 45));

  // masks: the 160x160 proto grid covers the letterboxed 640 square; crop out the padding
  if (protos) {
    const k = protos.dims[3] / SIZE;
    ctx.save();
    ctx.globalAlpha = 0.45;
    ctx.imageSmoothingEnabled = true;
    for (const d of dets) {
      const m = buildMask(d, protos);
      ctx.drawImage(m, pre.px * k, pre.py * k, pre.nw * k, pre.nh * k, 0, 0, W, H);
    }
    ctx.restore();
  }
  const list = $("dets");
  list.innerHTML = "";
  for (const d of dets) {
    const x1 = (d.x1 - pre.px) / pre.r, y1 = (d.y1 - pre.py) / pre.r;
    const x2 = (d.x2 - pre.px) / pre.r, y2 = (d.y2 - pre.py) / pre.r;
    const col = COLORS[d.cls], label = `${NAMES[d.cls]} ${(d.score * 100).toFixed(0)}%`;
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const tw = ctx.measureText(label).width + 8, ty = Math.max(0, y1 - fs - 6);
    ctx.fillStyle = col; ctx.fillRect(x1, ty, tw, fs + 6);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x1 + 4, ty + fs);
    const li = document.createElement("li");
    li.innerHTML = `<span class="sw" style="background:${col}"></span>${NAMES[d.cls]}<span class="p">${(d.score * 100).toFixed(1)}%</span>`;
    list.appendChild(li);
  }
  if (!dets.length) list.innerHTML = "<li>ไม่พบวัตถุ (ลองลด confidence)</li>";
  setStatus(`${lastRaw.mode === "seg" ? "Polygon" : "Rectangle"} · พบ ${dets.length} วัตถุ · ${ms.toFixed(0)} ms`);
  $("dl").disabled = false;
}

function loadSrc(src) {
  const img = new Image();
  img.onload = () => { currentImg = img; run().catch(fail); };
  img.onerror = () => setStatus("เปิดรูปไม่ได้");
  img.src = src;
}
function loadFile(f) { if (f && f.type.startsWith("image/")) loadSrc(URL.createObjectURL(f)); }
function fail(e) { console.error(e); setStatus("ผิดพลาด: " + e.message); }

$("file").onchange = (e) => loadFile(e.target.files[0]);
$("cam").onchange = (e) => loadFile(e.target.files[0]);
const stage = $("stage");
stage.ondragover = (e) => { e.preventDefault(); stage.classList.add("drag"); };
stage.ondragleave = () => stage.classList.remove("drag");
stage.ondrop = (e) => { e.preventDefault(); stage.classList.remove("drag"); loadFile(e.dataTransfer.files[0]); };
document.onpaste = (e) => { const it = [...e.clipboardData.items].find((i) => i.type.startsWith("image/")); if (it) loadFile(it.getAsFile()); };
$("conf").oninput = (e) => { $("confV").textContent = e.target.value; render(); };
$("iou").oninput = (e) => { $("iouV").textContent = e.target.value; render(); };
$("dl").onclick = () => { const a = document.createElement("a"); a.download = "prediction.png"; a.href = cv.toDataURL("image/png"); a.click(); };
for (const b of document.querySelectorAll("#modelSel button")) {
  b.onclick = () => {
    document.querySelectorAll("#modelSel button").forEach((x) => x.classList.toggle("on", x === b));
    mode = b.dataset.m;
    if (currentImg) run().catch(fail); else getSession(mode).then(() => setStatus("พร้อมใช้งาน")).catch(fail);
  };
}
for (const s of SAMPLES) {
  const t = document.createElement("img");
  t.src = s; t.alt = "sample"; t.onclick = () => loadSrc(s);
  $("samples").appendChild(t);
}
getSession("seg").then(() => setStatus("พร้อมใช้งาน — เลือกรูปได้เลย")).catch(fail);
