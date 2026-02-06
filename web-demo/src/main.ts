import "./style.css";
import FFT from "fft.js";
import * as ort from "onnxruntime-web";

const SR = 32000;
const N_FFT = 1024;
const HOP = 256;
const N_MELS = 128;
const MAX_TABLE_ROWS = 5000;

const MODEL_URL_DEFAULT =
  "/assets/BirdNET+_V3.0-preview3_Global_11K_FP32.onnx";
const LABELS_URL_DEFAULT =
  "/assets/BirdNET+_V3.0-preview3_Global_11K_Labels.csv";

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T | null;

const elements = {
  loadModelBtn: $("load-model") as HTMLButtonElement,
  loadScreen: $("load-screen") as HTMLDivElement,
  loadProgress: $("load-progress") as HTMLDivElement,
  loadStatus: $("load-status") as HTMLParagraphElement,
  appShell: $("app-shell") as HTMLDivElement,
  modelStatus: $("model-status") as HTMLSpanElement,
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
  downloadCsv: $("download-csv") as HTMLButtonElement
};

if (!elements.loadModelBtn) {
  throw new Error("Required DOM elements missing");
}

let audioData: Float32Array | null = null;
let audioDuration = 0;
let labels: string[] = [];
let session: ort.InferenceSession | null = null;
let modelReady = false;
let lastDetectionsCsv = "";

ort.env.wasm.wasmPaths = "/ort/";

function setStatus(text: string) {
  elements.runStatus.textContent = text;
}

function setLoadStatus(text: string) {
  elements.loadStatus.textContent = text;
}

function setModelStatus(text: string, ok: boolean) {
  elements.modelStatus.textContent = text;
  elements.modelStatus.classList.toggle("status-pill--ok", ok);
  elements.modelStatus.classList.toggle("status-pill--bad", !ok);
}

function setLoadingState(loading: boolean) {
  elements.loadProgress.classList.toggle("hidden", !loading);
  elements.loadModelBtn.disabled = loading;
}

function showAppShell() {
  elements.loadScreen.classList.add("hidden");
  elements.appShell.classList.remove("hidden");
}

function bytesToSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

function chunkAudio(
  y: Float32Array,
  chunkLengthSec: number,
  overlapSec: number,
  sr: number
) {
  const chunkSamples = Math.max(1, Math.round(chunkLengthSec * sr));
  const overlapSamples = Math.round(overlapSec * sr);
  if (overlapSamples >= chunkSamples) {
    throw new Error("Overlap must be smaller than chunk length.");
  }
  const step = chunkSamples - overlapSamples;
  const chunks: Float32Array[] = [];
  const spans: Array<[number, number]> = [];
  for (let start = 0; start < y.length; start += step) {
    const end = Math.min(start + chunkSamples, y.length);
    const slice = new Float32Array(chunkSamples);
    slice.set(y.slice(start, end));
    chunks.push(slice);
    spans.push([start / sr, end / sr]);
    if (end >= y.length) break;
  }
  return { chunks, spans, chunkSamples };
}

function parseCsvLine(line: string, delimiter: string) {
  const out: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === delimiter && !inQuotes) {
      out.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  out.push(current);
  return out;
}

function parseLabelsCsv(text: string): string[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0], ";");
  const sciIndex = header.indexOf("sci_name");
  const comIndex = header.indexOf("com_name");
  const labels: string[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = parseCsvLine(lines[i], ";");
    const sci = (cols[sciIndex] || "").trim();
    const com = (cols[comIndex] || "").trim();
    if (sci || com) {
      labels.push(`${sci}_${com}`);
    }
  }
  return labels;
}


function renderDetectionsTable(rows: DetectionRow[]) {
  const tbody = elements.detectionsTable.querySelector("tbody") as HTMLTableSectionElement;
  tbody.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const limit = Math.min(rows.length, MAX_TABLE_ROWS);
  for (let i = 0; i < limit; i += 1) {
    const row = rows[i];
    const tr = document.createElement("tr");
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

async function loadLabelsFromUrl(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch labels: ${res.status}`);
  }
  const text = await res.text();
  const parsed = parseLabelsCsv(text);
  if (!parsed.length) {
    throw new Error("No labels parsed from CSV.");
  }
  return parsed;
}

async function loadModelFromUrl(url: string) {
  return ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });
}

async function loadModelFromFile(file: File) {
  const buffer = await file.arrayBuffer();
  return ort.InferenceSession.create(buffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });
}

async function loadLabelsFromFile(file: File) {
  const text = await file.text();
  const parsed = parseLabelsCsv(text);
  if (!parsed.length) {
    throw new Error("No labels parsed from CSV.");
  }
  return parsed;
}

function updateRunButton() {
  const ready = modelReady && !!audioData;
  elements.runInferenceBtn.disabled = !ready;
}

async function handleLoadModel() {
  try {
    setLoadingState(true);
    setLoadStatus("Loading model...");
    setModelStatus("Loading model...", false);
    const localModel = elements.localModel.files?.[0] || null;
    const localLabels = elements.localLabels.files?.[0] || null;
    if (localModel && localLabels) {
      labels = await loadLabelsFromFile(localLabels);
      session = await loadModelFromFile(localModel);
    } else {
      const modelUrl = elements.modelUrl.value || MODEL_URL_DEFAULT;
      const labelsUrl = elements.labelsUrl.value || LABELS_URL_DEFAULT;
      labels = await loadLabelsFromUrl(labelsUrl);
      session = await loadModelFromUrl(modelUrl);
    }
    modelReady = true;
    setModelStatus("Model ready", true);
    setLoadStatus("Model ready");
    showAppShell();
  } catch (err) {
    modelReady = false;
    session = null;
    setModelStatus("Failed to load model", false);
    setLoadStatus((err as Error).message);
  }
  setLoadingState(false);
  updateRunButton();
}

type DetectionRow = {
  start: number;
  end: number;
  scientific: string;
  common: string;
  confidence: number;
};

function pickPredictionTensor(outputs: Record<string, ort.Tensor>, labelCount: number) {
  const tensors = Object.values(outputs);
  const byLabelCount = tensors.find((t) => t.dims?.length === 2 && t.dims[1] === labelCount);
  if (byLabelCount) return byLabelCount;
  return tensors[0];
}

async function runInference() {
  if (!session || !audioData) return;
  const chunkLength = Number(elements.chunkLength.value);
  const overlap = Number(elements.overlap.value);
  const batchSize = Number(elements.batchSize.value);
  const minConf = Number(elements.minConf.value);

  setStatus("Chunking audio...");
  const { chunks, spans, chunkSamples } = chunkAudio(audioData, chunkLength, overlap, SR);
  if (!chunks.length) {
    setStatus("No audio samples to process.");
    return;
  }

  const labelCount = labels.length;

  const detections: DetectionRow[] = [];
  const startTime = performance.now();

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const batchCount = batch.length;
    const input = new Float32Array(batchCount * chunkSamples);
    for (let b = 0; b < batchCount; b += 1) {
      input.set(batch[b], b * chunkSamples);
    }
    const tensor = new ort.Tensor("float32", input, [batchCount, chunkSamples]);
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: tensor
    };
    setStatus(`Running inference... ${Math.min(i + batchCount, chunks.length)}/${chunks.length}`);
    const results = await session.run(feeds);
    const predTensor = pickPredictionTensor(results, labelCount);
    const preds = predTensor.data as Float32Array;
    const outDim = predTensor.dims?.[1] || labelCount;

    for (let b = 0; b < batchCount; b += 1) {
      const offset = b * outDim;
      const [start, end] = spans[i + b];
      for (let c = 0; c < labelCount; c += 1) {
        const val = preds[offset + c];
        if (val >= minConf) {
          const label = labels[c] || "";
          const parts = label.split("_");
          detections.push({
            start,
            end,
            scientific: parts[0] || "",
            common: parts[1] || "",
            confidence: val
          });
        }
      }
    }
  }

  const duration = ((performance.now() - startTime) / 1000).toFixed(2);
  setStatus(`Inference complete in ${duration}s. Audio duration ${audioDuration.toFixed(2)}s.`);

  detections.sort((a, b) => (a.start === b.start ? b.confidence - a.confidence : a.start - b.start));
  renderDetectionsTable(detections);

  elements.resultsSection.classList.remove("hidden");

  lastDetectionsCsv = buildDetectionsCsv(detections);
  elements.downloadCsv.disabled = detections.length === 0;
}

function renderSpectrogram(y: Float32Array) {
  const canvas = elements.spectrogram;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const spectrogram = computeMelSpectrogram(y);
  const height = spectrogram.length;
  const width = spectrogram[0]?.length || 0;
  const image = ctx.createImageData(width, height);
  let max = -Infinity;
  for (const row of spectrogram) {
    for (const v of row) {
      if (v > max) max = v;
    }
  }
  const min = max - 80;
  for (let yIdx = 0; yIdx < height; yIdx += 1) {
    for (let xIdx = 0; xIdx < width; xIdx += 1) {
      const value = spectrogram[yIdx][xIdx];
      const norm = Math.min(1, Math.max(0, (value - min) / (max - min)));
      const idx = (yIdx * width + xIdx) * 4;
      const color = colorMap(norm);
      image.data[idx] = color[0];
      image.data[idx + 1] = color[1];
      image.data[idx + 2] = color[2];
      image.data[idx + 3] = 255;
    }
  }
  canvas.width = width;
  canvas.height = height;
  ctx.putImageData(image, 0, 0);
}

function colorMap(t: number): [number, number, number] {
  const r = Math.round(40 + 200 * t);
  const g = Math.round(30 + 150 * t);
  const b = Math.round(80 + 120 * (1 - t));
  return [r, g, b];
}

function computeMelSpectrogram(y: Float32Array) {
  const fft = new FFT(N_FFT);
  const window = hannWindow(N_FFT);
  const melFilters = createMelFilterbank(N_MELS, N_FFT, SR, 0, SR / 2);
  const frames = Math.floor((y.length - N_FFT) / HOP) + 1;
  const spectrogram: number[][] = Array.from({ length: N_MELS }, () => new Array(frames).fill(0));

  const input = fft.createComplexArray();
  const output = fft.createComplexArray();

  for (let i = 0; i < frames; i += 1) {
    const offset = i * HOP;
    for (let j = 0; j < N_FFT; j += 1) {
      input[2 * j] = (y[offset + j] || 0) * window[j];
      input[2 * j + 1] = 0;
    }
    fft.transform(output, input);
    const power = new Float32Array(N_FFT / 2 + 1);
    for (let k = 0; k < power.length; k += 1) {
      const re = output[2 * k];
      const im = output[2 * k + 1];
      power[k] = re * re + im * im;
    }
    for (let m = 0; m < N_MELS; m += 1) {
      let sum = 0;
      const filter = melFilters[m];
      for (let k = 0; k < filter.length; k += 1) {
        sum += power[k] * filter[k];
      }
      spectrogram[m][i] = 10 * Math.log10(sum + 1e-10);
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

function hzToMel(hz: number) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function createMelFilterbank(mels: number, nFft: number, sr: number, fMin: number, fMax: number) {
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = new Float32Array(mels + 2);
  for (let i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melMin + (i / (mels + 1)) * (melMax - melMin);
  }
  const hzPoints = Array.from(melPoints, melToHz);
  const bins = hzPoints.map((hz) => Math.floor((nFft + 1) * hz / sr));
  const filters: Float32Array[] = [];
  const nFreqs = Math.floor(nFft / 2) + 1;

  for (let m = 1; m <= mels; m += 1) {
    const filter = new Float32Array(nFreqs);
    const left = bins[m - 1];
    const center = bins[m];
    const right = bins[m + 1];
    for (let k = left; k < center; k += 1) {
      if (k >= 0 && k < nFreqs) {
        filter[k] = (k - left) / Math.max(1, center - left);
      }
    }
    for (let k = center; k < right; k += 1) {
      if (k >= 0 && k < nFreqs) {
        filter[k] = (right - k) / Math.max(1, right - center);
      }
    }
    filters.push(filter);
  }
  return filters;
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
  resetDownloads();
  elements.resultsSection.classList.add("hidden");
  elements.spectrogramSection.classList.remove("hidden");
  updateRunButton();
  renderSpectrogram(audioData);
}

elements.loadModelBtn.addEventListener("click", () => {
  handleLoadModel();
});

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

elements.runInferenceBtn.addEventListener("click", () => {
  runInference();
});

elements.downloadCsv.addEventListener("click", () => {
  if (!lastDetectionsCsv) return;
  downloadText("detections.csv", lastDetectionsCsv);
});


updateRunButton();
setStatus("Upload audio to begin.");
setLoadStatus("Model not loaded");
