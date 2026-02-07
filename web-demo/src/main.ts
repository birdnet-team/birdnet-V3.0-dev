import "./style.css";
import FFT from "fft.js";
import InferenceWorker from "./inference.worker.ts?worker";

const SR = 32000;
const N_FFT = 2048;
const HOP = 512;
const MAX_TABLE_ROWS = 500;
const DYNAMIC_RANGE_DB = 80;

const MODEL_URL_DEFAULT =
  "/assets/BirdNET+_V3.0-preview3_Global_11K_FP32.onnx";
const LABELS_URL_DEFAULT =
  "/assets/BirdNET+_V3.0-preview3_Global_11K_Labels.csv";

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T | null;

const elements = {
  appShell: $("app-shell") as HTMLDivElement,
  modelStatus: $("model-status") as HTMLSpanElement,
  providerStatus: $("provider-status") as HTMLSpanElement,
  audioFile: $("audio-file") as HTMLInputElement,
  audioMeta: $("audio-meta") as HTMLSpanElement,
  dropzone: $("dropzone") as HTMLDivElement,
  spectrogram: $("spectrogram") as HTMLCanvasElement,
  spectrogramSection: $("spectrogram-section") as HTMLDivElement,
  resultsSection: $("results-section") as HTMLDivElement,
  chunkLength: $("chunk-length") as HTMLInputElement,
  overlap: $("overlap") as HTMLInputElement,
  batchSize: $("batch-size") as HTMLInputElement,
  minConf: $("min-conf") as HTMLInputElement,
  modelUrl: $("model-url") as HTMLInputElement,
  labelsUrl: $("labels-url") as HTMLInputElement,
  localModel: $("local-model") as HTMLInputElement,
  localLabels: $("local-labels") as HTMLInputElement,
  runInferenceBtn: $("run-inference") as HTMLButtonElement,
  runStatus: $("run-status") as HTMLParagraphElement,
  detectionsTable: $("detections-table") as HTMLTableElement,
  detectionsNote: $("detections-note") as HTMLParagraphElement,
  downloadCsv: $("download-csv") as HTMLButtonElement,
  audioControls: $("audio-controls") as HTMLDivElement,
  audioPlayer: $("audio-player") as HTMLAudioElement,
  audioTime: $("audio-time") as HTMLSpanElement,
  segmentStatus: $("segment-status") as HTMLSpanElement
};

if (!elements.runInferenceBtn) {
  throw new Error("Required DOM elements missing");
}

let audioData: Float32Array | null = null;
let audioDuration = 0;
let modelReady = false;
let modelLoading = false;
let lastDetectionsCsv = "";
let audioUrl: string | null = null;
let segmentEnd: number | null = null;
let activeSegmentRow: HTMLTableRowElement | null = null;
let workerMsgId = 0;

// Initialize web worker for inference
const inferenceWorker = new InferenceWorker();
const pendingWorkerCalls = new Map<number, {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  onProgress?: (current: number, total: number) => void;
}>();

inferenceWorker.onmessage = (event: MessageEvent) => {
  const msg = event.data;
  if (msg.type === "inferenceProgress" && pendingWorkerCalls.has(msg.id)) {
    const call = pendingWorkerCalls.get(msg.id)!;
    call.onProgress?.(msg.current, msg.total);
    return;
  }
  if (pendingWorkerCalls.has(msg.id)) {
    const call = pendingWorkerCalls.get(msg.id)!;
    pendingWorkerCalls.delete(msg.id);
    call.resolve(msg);
  }
};

inferenceWorker.onerror = (err: ErrorEvent) => {
  console.error("Worker error:", err);
};

function callWorker<T>(
  message: Record<string, unknown>,
  onProgress?: (current: number, total: number) => void
): Promise<T> {
  const id = ++workerMsgId;
  return new Promise((resolve, reject) => {
    pendingWorkerCalls.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
    inferenceWorker.postMessage({ ...message, id });
  });
}

function setStatus(text: string) {
  elements.runStatus.textContent = text;
}

function setSegmentStatus(text: string) {
  elements.segmentStatus.textContent = text;
}

function setModelStatus(text: string, ok: boolean) {
  elements.modelStatus.textContent = text;
  elements.modelStatus.classList.toggle("status-pill--ok", ok);
  elements.modelStatus.classList.toggle("status-pill--bad", !ok);
}

function setProviderStatus(text: string, level: "ok" | "warn") {
  elements.providerStatus.textContent = text;
  elements.providerStatus.classList.toggle("status-pill--ok", level === "ok");
  elements.providerStatus.classList.toggle("status-pill--warn", level === "warn");
}

function bytesToSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00";
  const total = Math.floor(seconds);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function updateAudioTime() {
  const current = elements.audioPlayer.currentTime || 0;
  const duration = elements.audioPlayer.duration || audioDuration || 0;
  elements.audioTime.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
}

function clearSegmentPlayback() {
  segmentEnd = null;
  setSegmentStatus("");
  if (activeSegmentRow) {
    activeSegmentRow.classList.remove("table-row--active");
    activeSegmentRow = null;
  }
}

function playSegment(start: number, end: number, row?: HTMLTableRowElement) {
  if (!elements.audioPlayer.src) return;
  const clampedStart = Math.max(0, start);
  const clampedEnd = Math.max(clampedStart, end);
  segmentEnd = clampedEnd;
  elements.audioPlayer.currentTime = clampedStart;
  elements.audioPlayer.play();
  setSegmentStatus(`Playing ${clampedStart.toFixed(2)}s - ${clampedEnd.toFixed(2)}s`);
  if (activeSegmentRow && activeSegmentRow !== row) {
    activeSegmentRow.classList.remove("table-row--active");
  }
  if (row) {
    row.classList.add("table-row--active");
    activeSegmentRow = row;
  }
}


async function decodeAudio(file: File): Promise<Float32Array> {
  const arrayBuffer = await file.arrayBuffer();
  const context = new AudioContext();
  const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
  const channelData = mixToMono(buffer);
  const resampled = resampleLinear(channelData, buffer.sampleRate, SR);
  audioDuration = resampled.length / SR;
  await context.close();
  return resampled;
}

function mixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0).slice();
  }
  const length = buffer.length;
  const mix = new Float32Array(length);
  for (let c = 0; c < buffer.numberOfChannels; c += 1) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i += 1) {
      mix[i] += data[i] / buffer.numberOfChannels;
    }
  }
  return mix;
}

function resampleLinear(input: Float32Array, srcSr: number, targetSr: number) {
  if (srcSr === targetSr) return input;
  const ratio = srcSr / targetSr;
  const newLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(newLength);
  for (let i = 0; i < newLength; i += 1) {
    const srcIndex = i * ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = srcIndex - i0;
    output[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return output;
}


function renderDetectionsTable(rows: DetectionRow[]) {
  const tbody = elements.detectionsTable.querySelector("tbody") as HTMLTableSectionElement;
  tbody.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const limit = Math.min(rows.length, MAX_TABLE_ROWS);
  for (let i = 0; i < limit; i += 1) {
    const row = rows[i];
    const tr = document.createElement("tr");
    tr.classList.add("table-row");
    tr.tabIndex = 0;
    tr.setAttribute("role", "button");
    tr.setAttribute("aria-label", `Play ${row.start.toFixed(2)} to ${row.end.toFixed(2)} seconds`);
    tr.dataset.start = row.start.toString();
    tr.dataset.end = row.end.toString();
    tr.innerHTML = `
      <td>${row.start.toFixed(3)}</td>
      <td>${row.end.toFixed(3)}</td>
      <td>${row.scientific}</td>
      <td>${row.common}</td>
      <td>${row.confidence.toFixed(3)}</td>
    `;
    fragment.appendChild(tr);
  }
  tbody.appendChild(fragment);
  if (rows.length > MAX_TABLE_ROWS) {
    elements.detectionsNote.textContent =
      `Showing ${MAX_TABLE_ROWS} of ${rows.length} detections. ` +
      "Increase min confidence to reduce rows.";
  } else {
    elements.detectionsNote.textContent = rows.length
      ? `${rows.length} detections`
      : "No detections above threshold.";
  }
}

function buildDetectionsCsv(rows: DetectionRow[]) {
  const header = ["start_sec", "end_sec", "scientific_name", "common_name", "confidence"];
  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push(
      [
        row.start.toFixed(3),
        row.end.toFixed(3),
        row.scientific,
        row.common,
        row.confidence.toFixed(6)
      ].join(",")
    );
  }
  return lines.join("\n");
}


function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function updateRunButton() {
  const ready = !!audioData && !modelLoading;
  elements.runInferenceBtn.disabled = !ready;
}

async function handleLoadModel() {
  if (modelLoading) return;
  try {
    modelLoading = true;
    updateRunButton();
    setStatus("Loading model...");
    setModelStatus("Loading model...", false);

    const localModel = elements.localModel.files?.[0] || null;
    const localLabels = elements.localLabels.files?.[0] || null;

    type LoadModelResult = {
      type: string;
      success: boolean;
      provider?: "webgl" | "wasm";
      labelCount?: number;
      error?: string;
    };

    let result: LoadModelResult;
    if (localModel && localLabels) {
      const modelBuffer = await localModel.arrayBuffer();
      const labelsText = await localLabels.text();
      result = await callWorker<LoadModelResult>({
        type: "loadModel",
        modelBuffer,
        labelsText
      });
    } else {
      const modelUrl = elements.modelUrl.value || MODEL_URL_DEFAULT;
      const labelsUrl = elements.labelsUrl.value || LABELS_URL_DEFAULT;
      result = await callWorker<LoadModelResult>({
        type: "loadModel",
        modelUrl,
        labelsUrl
      });
    }

    if (result.success) {
      modelReady = true;
      setModelStatus("Model ready", true);
      if (result.provider === "webgl") {
        setProviderStatus("Provider: WebGL", "ok");
      } else {
        setProviderStatus("Provider: WASM (WebGL unavailable)", "warn");
      }
    } else {
      throw new Error(result.error || "Failed to load model");
    }
  } catch (err) {
    modelReady = false;
    setModelStatus("Failed to load model", false);
    setStatus((err as Error).message);
  }
  modelLoading = false;
  updateRunButton();
}

type DetectionRow = {
  start: number;
  end: number;
  scientific: string;
  common: string;
  confidence: number;
};

async function runInference() {
  if (!modelReady || !audioData) return;

  const chunkLength = Number(elements.chunkLength.value);
  const overlap = Number(elements.overlap.value);
  const batchSize = Number(elements.batchSize.value);
  const minConf = Number(elements.minConf.value);

  setStatus("Running inference...");

  type InferenceResult = {
    type: string;
    success: boolean;
    detections?: DetectionRow[];
    duration?: number;
    error?: string;
  };

  const result = await callWorker<InferenceResult>(
    {
      type: "runInference",
      audioData,
      chunkLength,
      overlap,
      batchSize,
      minConf,
      sampleRate: SR
    },
    (current, total) => {
      setStatus(`Running inference... ${current}/${total}`);
    }
  );

  if (!result.success) {
    setStatus(result.error || "Inference failed");
    return;
  }

  const detections = result.detections || [];
  const duration = result.duration?.toFixed(2) || "0";
  setStatus(`Inference complete in ${duration}s. Audio duration ${audioDuration.toFixed(2)}s.`);

  renderDetectionsTable(detections);
  elements.resultsSection.classList.remove("hidden");

  lastDetectionsCsv = buildDetectionsCsv(detections);
  elements.downloadCsv.disabled = detections.length === 0;
}

function renderSpectrogram(y: Float32Array) {
  const canvas = elements.spectrogram;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const spectrogram = computeLinearSpectrogram(y);
  const height = spectrogram.length;
  const width = spectrogram[0]?.length || 0;
  if (width === 0) return;

  // Gather all values for percentile-based normalization
  const allVals: number[] = [];
  for (const row of spectrogram) {
    for (const v of row) {
      allVals.push(v);
    }
  }
  allVals.sort((a, b) => a - b);

  // Use percentiles for robust min/max (ignore outliers)
  const pLow = Math.floor(allVals.length * 0.02);
  const pHigh = Math.floor(allVals.length * 0.98);
  const minDb = allVals[pLow];
  const maxDb = allVals[pHigh];
  const rangeDb = Math.max(maxDb - minDb, DYNAMIC_RANGE_DB);

  // Create image at native resolution then scale up
  const nativeImage = ctx.createImageData(width, height);
  for (let yIdx = 0; yIdx < height; yIdx += 1) {
    const srcRow = height - 1 - yIdx; // Flip vertically
    for (let xIdx = 0; xIdx < width; xIdx += 1) {
      const value = spectrogram[srcRow][xIdx];
      let norm = (value - minDb) / rangeDb;
      norm = Math.min(1, Math.max(0, norm));
      // Gamma correction for better mid-tone visibility
      norm = Math.pow(norm, 0.75);
      const idx = (yIdx * width + xIdx) * 4;
      const color = colorMap(norm);
      nativeImage.data[idx] = color[0];
      nativeImage.data[idx + 1] = color[1];
      nativeImage.data[idx + 2] = color[2];
      nativeImage.data[idx + 3] = 255;
    }
  }

  // Render at fixed display size with smooth scaling
  const displayWidth = 900;
  const displayHeight = 200;
  canvas.width = displayWidth;
  canvas.height = displayHeight;

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = width;
  tmpCanvas.height = height;
  tmpCanvas.getContext("2d")!.putImageData(nativeImage, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmpCanvas, 0, 0, displayWidth, displayHeight);
}

function colorMap(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const stops = [
    { t: 0.0, c: [68, 1, 84] },
    { t: 0.13, c: [71, 44, 122] },
    { t: 0.25, c: [59, 81, 139] },
    { t: 0.38, c: [44, 113, 142] },
    { t: 0.5, c: [33, 145, 140] },
    { t: 0.63, c: [39, 173, 129] },
    { t: 0.75, c: [92, 200, 99] },
    { t: 0.88, c: [170, 220, 50] },
    { t: 1.0, c: [253, 231, 37] }
  ];
  for (let i = 0; i < stops.length - 1; i += 1) {
    const a = stops[i];
    const b = stops[i + 1];
    if (clamped >= a.t && clamped <= b.t) {
      const local = (clamped - a.t) / (b.t - a.t);
      return [
        Math.round(a.c[0] + (b.c[0] - a.c[0]) * local),
        Math.round(a.c[1] + (b.c[1] - a.c[1]) * local),
        Math.round(a.c[2] + (b.c[2] - a.c[2]) * local)
      ];
    }
  }
  const last = stops[stops.length - 1].c;
  return [last[0], last[1], last[2]];
}

function computeLinearSpectrogram(y: Float32Array) {
  // Pre-emphasis to boost high frequencies (common for speech/bird sounds)
  const preEmph = 0.97;
  const emphasized = new Float32Array(y.length);
  emphasized[0] = y[0];
  for (let i = 1; i < y.length; i += 1) {
    emphasized[i] = y[i] - preEmph * y[i - 1];
  }

  const fft = new FFT(N_FFT);
  const window = hannWindow(N_FFT);
  const frames = Math.floor((emphasized.length - N_FFT) / HOP) + 1;
  const nBins = Math.floor(N_FFT / 2) + 1;
  const spectrogram: number[][] = Array.from({ length: nBins }, () => new Array(frames).fill(0));
  const eps = 1e-10;

  const input = fft.createComplexArray();
  const output = fft.createComplexArray();

  for (let i = 0; i < frames; i += 1) {
    const offset = i * HOP;
    for (let j = 0; j < N_FFT; j += 1) {
      input[2 * j] = (emphasized[offset + j] || 0) * window[j];
      input[2 * j + 1] = 0;
    }
    fft.transform(output, input);
    for (let k = 0; k < nBins; k += 1) {
      const re = output[2 * k];
      const im = output[2 * k + 1];
      const power = re * re + im * im;
      spectrogram[k][i] = 10 * Math.log10(power + eps); // Power spectrum in dB
    }
  }
  return spectrogram;
}

function hannWindow(length: number) {
  const win = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return win;
}


function resetDownloads() {
  elements.downloadCsv.disabled = true;
}

async function handleAudioFile(file?: File) {
  const targetFile = file ?? elements.audioFile.files?.[0];
  if (!targetFile) return;
  elements.audioMeta.textContent = `${targetFile.name} (${bytesToSize(targetFile.size)})`;
  setStatus("Decoding audio...");
  audioData = await decodeAudio(targetFile);
  setStatus(`Audio ready: ${audioDuration.toFixed(2)}s at ${SR} Hz.`);
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }
  audioUrl = URL.createObjectURL(targetFile);
  elements.audioPlayer.src = audioUrl;
  elements.audioPlayer.load();
  updateAudioTime();
  clearSegmentPlayback();
  resetDownloads();
  elements.resultsSection.classList.add("hidden");
  elements.spectrogramSection.classList.remove("hidden");
  elements.audioControls.classList.remove("hidden");
  updateRunButton();
  renderSpectrogram(audioData);
}

elements.dropzone.addEventListener("click", () => {
  elements.audioFile.click();
});

elements.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropzone.classList.add("dragover");
});

elements.dropzone.addEventListener("dragleave", () => {
  elements.dropzone.classList.remove("dragover");
});

elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("dragover");
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    handleAudioFile(file);
  }
});

elements.audioFile.addEventListener("change", () => {
  handleAudioFile();
});

elements.runInferenceBtn.addEventListener("click", async () => {
  if (!modelReady) {
    elements.runInferenceBtn.classList.add("btn--loading");
    elements.runInferenceBtn.disabled = true;
    await handleLoadModel();
  }
  if (!modelReady) {
    elements.runInferenceBtn.classList.remove("btn--loading");
    elements.runInferenceBtn.disabled = false;
    return;
  }
  elements.runInferenceBtn.classList.add("btn--loading");
  elements.runInferenceBtn.disabled = true;
  try {
    await runInference();
  } finally {
    elements.runInferenceBtn.classList.remove("btn--loading");
    elements.runInferenceBtn.disabled = false;
  }
});

elements.downloadCsv.addEventListener("click", () => {
  if (!lastDetectionsCsv) return;
  downloadText("detections.csv", lastDetectionsCsv);
});

elements.audioPlayer.addEventListener("loadedmetadata", () => {
  updateAudioTime();
});

elements.audioPlayer.addEventListener("timeupdate", () => {
  updateAudioTime();
  if (segmentEnd !== null && elements.audioPlayer.currentTime >= segmentEnd) {
    elements.audioPlayer.pause();
    clearSegmentPlayback();
  }
});

elements.audioPlayer.addEventListener("pause", () => {
  if (segmentEnd !== null && elements.audioPlayer.currentTime < segmentEnd) {
    clearSegmentPlayback();
  }
});

elements.detectionsTable.addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  const row = target.closest("tr");
  if (!row || !(row instanceof HTMLTableRowElement)) return;
  const start = Number(row.dataset.start);
  const end = Number(row.dataset.end);
  if (Number.isFinite(start) && Number.isFinite(end)) {
    playSegment(start, end, row);
  }
});

elements.detectionsTable.addEventListener("keydown", (event) => {
  if (!(event.key === "Enter" || event.key === " ")) return;
  const target = event.target as HTMLElement;
  const row = target.closest("tr");
  if (!row || !(row instanceof HTMLTableRowElement)) return;
  const start = Number(row.dataset.start);
  const end = Number(row.dataset.end);
  if (Number.isFinite(start) && Number.isFinite(end)) {
    event.preventDefault();
    playSegment(start, end, row);
  }
});


updateRunButton();
setStatus("Upload audio to begin.");
setModelStatus("Model not loaded", false);
