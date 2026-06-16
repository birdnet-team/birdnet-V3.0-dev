/**
 * BirdNET+ Web Demo - Main Application
 *
 * This is the entry point for the BirdNET+ web-based bird sound identification demo.
 * It handles:
 * - Audio file upload and decoding
 * - Spectrogram visualization
 * - Model loading and inference (via web worker)
 * - Results display and export
 *
 * Architecture:
 * - Audio processing and spectrogram: runs on main thread
 * - Model inference: offloaded to web worker to prevent UI freezing
 */

import "./style.css";

// Local modules
import { SAMPLE_RATE } from "./constants";
import { decodeAudioFile, formatTime } from "./audio";
import { renderSpectrogram } from "./spectrogram";
import type { DetectionRow, WorkerLoadModelResult, WorkerInferenceResult } from "./types";

// Web worker for inference (loaded via Vite's worker import)
import InferenceWorker from "./inference.worker.ts?worker";

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/** Model base name (without precision suffix) */
const MODEL_BASE = "/assets/BirdNET+_V3.0-preview3.1_Global_11K";
const LABELS_URL_DEFAULT = "/assets/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv";

/** Get model URL for a given format */
function getModelUrl(format: string): string {
  return `${MODEL_BASE}_${format}.onnx`;
}

/** Adds a cache-busting timestamp while preserving existing query parameters. */
function addCacheBuster(url: string): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${Date.now()}`;
}

/** Maximum rows to display in detection table (for performance) */
const MAX_TABLE_ROWS = 500;

// -----------------------------------------------------------------------------
// DOM Elements
// -----------------------------------------------------------------------------

/**
 * Helper to safely get DOM element by ID with type casting.
 * Returns null if element not found (caller should handle).
 */
const $ = <T extends HTMLElement>(id: string): T | null =>
  document.getElementById(id) as T | null;

/**
 * All DOM elements used by the application.
 * Centralized here for easy reference and type safety.
 */
const elements = {
  // Status indicators
  modelStatus: $<HTMLSpanElement>("model-status")!,
  providerStatus: $<HTMLSpanElement>("provider-status")!,
  runStatus: $<HTMLParagraphElement>("run-status")!,
  segmentStatus: $<HTMLSpanElement>("segment-status")!,

  // Audio input
  audioFile: $<HTMLInputElement>("audio-file")!,
  audioMeta: $<HTMLSpanElement>("audio-meta")!,
  dropzone: $<HTMLDivElement>("dropzone")!,
  fileInfoBar: $<HTMLDivElement>("file-info-bar")!,
  fileInfoName: $<HTMLSpanElement>("file-info-name")!,
  fileInfoSize: $<HTMLSpanElement>("file-info-size")!,
  changeFileBtn: $<HTMLButtonElement>("change-file-btn")!,

  // Visualization
  spectrogram: $<HTMLCanvasElement>("spectrogram")!,
  spectrogramSection: $<HTMLDivElement>("spectrogram-section")!,

  // Audio playback
  audioControls: $<HTMLDivElement>("audio-controls")!,
  audioPlayer: $<HTMLAudioElement>("audio-player")!,
  audioTime: $<HTMLSpanElement>("audio-time")!,

  // Model settings
  modelFormat: $<HTMLSelectElement>("model-format")!,
  chunkLength: $<HTMLInputElement>("chunk-length")!,
  overlap: $<HTMLInputElement>("overlap")!,
  batchSize: $<HTMLInputElement>("batch-size")!,
  minConf: $<HTMLInputElement>("min-conf")!,
  modelUrl: $<HTMLInputElement>("model-url")!,
  labelsUrl: $<HTMLInputElement>("labels-url")!,
  localModel: $<HTMLInputElement>("local-model")!,
  localLabels: $<HTMLInputElement>("local-labels")!,

  // Results
  resultsSection: $<HTMLDivElement>("results-section")!,
  detectionsTable: $<HTMLTableElement>("detections-table")!,
  detectionsNote: $<HTMLParagraphElement>("detections-note")!,

  // Action buttons
  runInferenceBtn: $<HTMLButtonElement>("run-inference")!,
  downloadCsv: $<HTMLButtonElement>("download-csv")!
};

// Verify critical elements exist
if (!elements.runInferenceBtn || !elements.dropzone) {
  throw new Error("Required DOM elements missing - check HTML structure");
}

// -----------------------------------------------------------------------------
// Application State
// -----------------------------------------------------------------------------

/** Current audio data (mono Float32Array at 32kHz) */
let audioData: Float32Array | null = null;

/** Duration of current audio in seconds */
let audioDuration = 0;

/** Object URL for current audio file (for playback) */
let audioObjectUrl: string | null = null;

/** Whether the ONNX model is loaded and ready */
let modelReady = false;

/** Whether model is currently being loaded */
let modelLoading = false;

/** Last generated CSV for download */
let lastDetectionsCsv = "";

/** End time for segment playback (null if not playing segment) */
let segmentEndTime: number | null = null;

/** Currently highlighted table row during segment playback */
let activeSegmentRow: HTMLTableRowElement | null = null;

// -----------------------------------------------------------------------------
// Web Worker Communication
// -----------------------------------------------------------------------------

/** Counter for generating unique message IDs */
let workerMessageId = 0;

/** Initialize the inference worker */
const inferenceWorker = new InferenceWorker();

/**
 * Pending worker calls waiting for responses.
 * Maps message ID to resolve/reject callbacks.
 */
const pendingWorkerCalls = new Map<
  number,
  {
    resolve: (value: unknown) => void;
    reject: (reason: unknown) => void;
    onProgress?: (current: number, total: number) => void;
  }
>();

// Handle messages from worker
inferenceWorker.onmessage = (event: MessageEvent) => {
  const msg = event.data;

  // Handle progress updates (don't resolve promise yet)
  if (msg.type === "inferenceProgress" && pendingWorkerCalls.has(msg.id)) {
    const call = pendingWorkerCalls.get(msg.id)!;
    call.onProgress?.(msg.current, msg.total);
    return;
  }

  // Handle final responses
  if (pendingWorkerCalls.has(msg.id)) {
    const call = pendingWorkerCalls.get(msg.id)!;
    pendingWorkerCalls.delete(msg.id);
    call.resolve(msg);
  }
};

// Log worker errors for debugging
inferenceWorker.onerror = (err: ErrorEvent) => {
  console.error("Inference worker error:", err);
};

/**
 * Sends a message to the worker and returns a promise that resolves
 * when the worker responds.
 *
 * @param message - Message payload to send
 * @param onProgress - Optional callback for progress updates
 * @returns Promise resolving to worker response
 */
function callWorker<T>(
  message: Record<string, unknown>,
  onProgress?: (current: number, total: number) => void
): Promise<T> {
  const id = ++workerMessageId;
  return new Promise((resolve, reject) => {
    pendingWorkerCalls.set(id, {
      resolve: resolve as (v: unknown) => void,
      reject,
      onProgress
    });
    inferenceWorker.postMessage({ ...message, id });
  });
}

// -----------------------------------------------------------------------------
// UI Updates
// -----------------------------------------------------------------------------

/** Updates the main status message below the run button */
function setStatus(text: string): void {
  elements.runStatus.textContent = text;
}

/** Updates the segment playback status text */
function setSegmentStatus(text: string): void {
  elements.segmentStatus.textContent = text;
}

/** Updates the model status pill (loaded/not loaded) */
function setModelStatus(text: string, isOk: boolean): void {
  elements.modelStatus.textContent = text;
  elements.modelStatus.classList.toggle("status-pill-ok", isOk);
  elements.modelStatus.classList.toggle("status-pill-bad", !isOk);
}

/** Updates the execution provider status pill (WebGL/WASM) */
function setProviderStatus(text: string, level: "ok" | "warn"): void {
  elements.providerStatus.textContent = text;
  elements.providerStatus.classList.toggle("status-pill-ok", level === "ok");
  elements.providerStatus.classList.toggle("status-pill-warn", level === "warn");
}

/** Formats bytes as human-readable size (KB, MB) */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Updates the audio time display (current / total) */
function updateAudioTimeDisplay(): void {
  const current = elements.audioPlayer.currentTime || 0;
  const total = elements.audioPlayer.duration || audioDuration || 0;
  elements.audioTime.textContent = `${formatTime(current)} / ${formatTime(total)}`;
}

/** Enables/disables the run button based on state */
function updateRunButton(): void {
  const canRun = !!audioData && !modelLoading;
  elements.runInferenceBtn.disabled = !canRun;
}

// -----------------------------------------------------------------------------
// Segment Playback
// -----------------------------------------------------------------------------

/** Clears segment playback state and resets UI */
function clearSegmentPlayback(): void {
  segmentEndTime = null;
  setSegmentStatus("");
  if (activeSegmentRow) {
    activeSegmentRow.classList.remove("table-row-active");
    activeSegmentRow = null;
  }
}

/**
 * Plays a specific time segment of the audio.
 * Used when clicking on detection table rows.
 *
 * @param start - Start time in seconds
 * @param end - End time in seconds
 * @param row - Optional table row to highlight
 */
function playSegment(
  start: number,
  end: number,
  row?: HTMLTableRowElement
): void {
  if (!elements.audioPlayer.src) return;

  const clampedStart = Math.max(0, start);
  const clampedEnd = Math.max(clampedStart, end);

  segmentEndTime = clampedEnd;
  elements.audioPlayer.currentTime = clampedStart;
  elements.audioPlayer.play();

  setSegmentStatus(`Playing ${clampedStart.toFixed(2)}s - ${clampedEnd.toFixed(2)}s`);

  // Update row highlighting
  if (activeSegmentRow && activeSegmentRow !== row) {
    activeSegmentRow.classList.remove("table-row-active");
  }
  if (row) {
    row.classList.add("table-row-active");
    activeSegmentRow = row;
  }
}

// -----------------------------------------------------------------------------
// Results Table
// -----------------------------------------------------------------------------

/**
 * Renders detection results to the HTML table.
 * Limits rows for performance and updates count display.
 *
 * @param detections - Array of detection results to display
 */
function renderDetectionsTable(detections: DetectionRow[]): void {
  const tbody = elements.detectionsTable.querySelector("tbody");
  if (!tbody) return;

  tbody.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const limit = Math.min(detections.length, MAX_TABLE_ROWS);

  for (let i = 0; i < limit; i++) {
    const detection = detections[i];
    const tr = document.createElement("tr");
    tr.className = "cursor-pointer hover:bg-slate-50 transition-colors";
    tr.tabIndex = 0;
    tr.setAttribute("role", "button");
    tr.setAttribute(
      "aria-label",
      `Play ${detection.start.toFixed(2)} to ${detection.end.toFixed(2)} seconds`
    );
    tr.dataset.start = detection.start.toString();
    tr.dataset.end = detection.end.toString();
    tr.innerHTML = `
      <td class="p-2 border-b border-slate-100">${detection.start.toFixed(3)}</td>
      <td class="p-2 border-b border-slate-100">${detection.end.toFixed(3)}</td>
      <td class="p-2 border-b border-slate-100">${detection.scientific}</td>
      <td class="p-2 border-b border-slate-100">${detection.common}</td>
      <td class="p-2 border-b border-slate-100">${detection.confidence.toFixed(3)}</td>
    `;
    fragment.appendChild(tr);
  }

  tbody.appendChild(fragment);

  // Update note showing total count
  if (detections.length > MAX_TABLE_ROWS) {
    elements.detectionsNote.textContent =
      `Showing ${MAX_TABLE_ROWS} of ${detections.length} detections. ` +
      "Increase min confidence to reduce rows.";
  } else {
    elements.detectionsNote.textContent = detections.length
      ? `${detections.length} detections`
      : "No detections above threshold.";
  }
}

/**
 * Converts detection results to CSV format.
 *
 * @param detections - Array of detection results
 * @returns CSV string with header row
 */
function buildDetectionsCsv(detections: DetectionRow[]): string {
  const header = ["start_sec", "end_sec", "scientific_name", "common_name", "confidence"];
  const lines = [header.join(",")];

  for (const detection of detections) {
    lines.push(
      [
        detection.start.toFixed(3),
        detection.end.toFixed(3),
        detection.scientific,
        detection.common,
        detection.confidence.toFixed(6)
      ].join(",")
    );
  }

  return lines.join("\n");
}

/**
 * Triggers a file download with the given content.
 *
 * @param filename - Name for the downloaded file
 * @param content - File content as string
 */
function downloadTextFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

// -----------------------------------------------------------------------------
// Model Loading
// -----------------------------------------------------------------------------

/**
 * Loads the ONNX model and species labels into the inference worker.
 * Supports both remote URLs and local file uploads.
 */
async function loadModel(): Promise<void> {
  if (modelLoading) return;

  try {
    modelLoading = true;
    updateRunButton();
    setStatus("Loading model...");
    setModelStatus("Loading model...", false);

    // Check if user provided local files
    const localModel = elements.localModel.files?.[0] || null;
    const localLabels = elements.localLabels.files?.[0] || null;

    let result: WorkerLoadModelResult;

    if (localModel && localLabels) {
      // Load from local files
      const modelBuffer = await localModel.arrayBuffer();
      const labelsText = await localLabels.text();
      result = await callWorker<WorkerLoadModelResult>({
        type: "loadModel",
        modelBuffer,
        labelsText,
      });
    } else {
      // Load from URLs - use custom URL if provided, otherwise use format selector
      const customModelUrl = elements.modelUrl.value.trim();
      const modelUrl = addCacheBuster(customModelUrl || getModelUrl(elements.modelFormat.value));
      const labelsUrl = addCacheBuster(elements.labelsUrl.value || LABELS_URL_DEFAULT);
      result = await callWorker<WorkerLoadModelResult>({
        type: "loadModel",
        modelUrl,
        labelsUrl,
      });
    }

    if (result.success) {
      modelReady = true;
      setModelStatus("Model ready", true);

      // Show which execution provider was used
      if (result.provider === "wasm") {
        setProviderStatus("Provider: WASM (Recommended)", "ok");
      } else {
        setProviderStatus(`Provider: ${result.provider}`, "warn");
      }
    } else {
      throw new Error(result.error || "Failed to load model");
    }
  } catch (err) {
    modelReady = false;
    setModelStatus("Failed to load model", false);
    setStatus((err as Error).message);
  } finally {
    modelLoading = false;
    updateRunButton();
  }
}

// -----------------------------------------------------------------------------
// Inference
// -----------------------------------------------------------------------------

/**
 * Runs inference on the loaded audio data using parameters from UI.
 * Shows progress updates during processing.
 */
async function runInference(): Promise<void> {
  if (!modelReady || !audioData) return;

  // Get parameters from UI controls
  const chunkLength = Number(elements.chunkLength.value);
  const overlap = Number(elements.overlap.value);
  const batchSize = Number(elements.batchSize.value);
  const minConf = Number(elements.minConf.value);

  setStatus("Running inference...");

  // Call worker with progress callback
  const result = await callWorker<WorkerInferenceResult>(
    {
      type: "runInference",
      audioData,
      chunkLength,
      overlap,
      batchSize,
      minConf,
      sampleRate: SAMPLE_RATE
    },
    (current, total) => {
      setStatus(`Running inference... ${current}/${total}`);
    }
  );

  if (!result.success) {
    setStatus(result.error || "Inference failed");
    return;
  }

  // Show results
  const detections = result.detections || [];
  const duration = result.duration?.toFixed(2) || "0";
  setStatus(`Inference complete in ${duration}s. Audio duration ${audioDuration.toFixed(2)}s.`);

  renderDetectionsTable(detections);
  elements.resultsSection.classList.remove("hidden");

  // Prepare CSV for download
  lastDetectionsCsv = buildDetectionsCsv(detections);
  elements.downloadCsv.disabled = detections.length === 0;
}

// -----------------------------------------------------------------------------
// Audio File Handling
// -----------------------------------------------------------------------------

/**
 * Handles a newly selected audio file:
 * - Decodes audio to Float32Array
 * - Renders spectrogram
 * - Sets up audio player for playback
 *
 * @param file - Audio file to process (or uses file input value)
 */
async function handleAudioFile(file?: File): Promise<void> {
  const targetFile = file ?? elements.audioFile.files?.[0];
  if (!targetFile) return;

  // Show file info in the collapsed bar
  elements.fileInfoName.textContent = targetFile.name;
  elements.fileInfoSize.textContent = `(${formatBytes(targetFile.size)})`;
  elements.audioMeta.textContent = `${targetFile.name} (${formatBytes(targetFile.size)})`;
  
  // Collapse dropzone and show file info bar
  elements.dropzone.classList.add("hidden");
  elements.fileInfoBar.classList.remove("hidden");
  
  setStatus("Decoding audio...");

  // Decode to 32kHz mono
  audioData = await decodeAudioFile(targetFile, SAMPLE_RATE);
  audioDuration = audioData.length / SAMPLE_RATE;
  setStatus(`Audio ready: ${audioDuration.toFixed(2)}s at ${SAMPLE_RATE} Hz.`);

  // Set up audio player with original file for accurate playback
  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);
  }
  audioObjectUrl = URL.createObjectURL(targetFile);
  elements.audioPlayer.src = audioObjectUrl;
  elements.audioPlayer.load();
  updateAudioTimeDisplay();

  // Reset state
  clearSegmentPlayback();
  elements.downloadCsv.disabled = true;
  elements.resultsSection.classList.add("hidden");

  // Show spectrogram and audio controls
  elements.spectrogramSection.classList.remove("hidden");
  elements.audioControls.classList.remove("hidden");
  updateRunButton();

  // Render spectrogram visualization
  renderSpectrogram(elements.spectrogram, audioData);
}

// -----------------------------------------------------------------------------
// Event Listeners
// -----------------------------------------------------------------------------

// Dropzone: click to open file picker
elements.dropzone.addEventListener("click", () => {
  elements.audioFile.click();
});

// Model format change: reset model so it reloads on next run
elements.modelFormat.addEventListener("change", () => {
  if (modelReady) {
    modelReady = false;
    setModelStatus("Format changed - click Run to load", false);
    setStatus("Model format changed. Click Run Inference to load the new model.");
  }
});

// Dropzone: visual feedback on drag
elements.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropzone.classList.add("dragover");
});

elements.dropzone.addEventListener("dragleave", () => {
  elements.dropzone.classList.remove("dragover");
});

// Dropzone: handle file drop
elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("dragover");
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    handleAudioFile(file);
  }
});

// File input: handle selection via dialog
elements.audioFile.addEventListener("change", () => {
  handleAudioFile();
});

// Change file button: show dropzone again
elements.changeFileBtn.addEventListener("click", () => {
  elements.fileInfoBar.classList.add("hidden");
  elements.dropzone.classList.remove("hidden");
  elements.audioFile.click();
});

// Run button: load model (if needed) then run inference
elements.runInferenceBtn.addEventListener("click", async () => {
  // Load model on first click
  if (!modelReady) {
    elements.runInferenceBtn.classList.add("btn-loading");
    elements.runInferenceBtn.disabled = true;
    await loadModel();
  }

  // Abort if model failed to load
  if (!modelReady) {
    elements.runInferenceBtn.classList.remove("btn-loading");
    elements.runInferenceBtn.disabled = false;
    return;
  }

  // Run inference with loading state
  elements.runInferenceBtn.classList.add("btn-loading");
  elements.runInferenceBtn.disabled = true;
  try {
    await runInference();
  } finally {
    elements.runInferenceBtn.classList.remove("btn-loading");
    elements.runInferenceBtn.disabled = false;
  }
});

// Download CSV button
elements.downloadCsv.addEventListener("click", () => {
  if (lastDetectionsCsv) {
    downloadTextFile("detections.csv", lastDetectionsCsv);
  }
});

// Audio player: update time display
elements.audioPlayer.addEventListener("loadedmetadata", updateAudioTimeDisplay);
elements.audioPlayer.addEventListener("timeupdate", () => {
  updateAudioTimeDisplay();

  // Auto-pause at segment end
  if (segmentEndTime !== null && elements.audioPlayer.currentTime >= segmentEndTime) {
    elements.audioPlayer.pause();
    clearSegmentPlayback();
  }
});

// Audio player: handle manual pause during segment playback
elements.audioPlayer.addEventListener("pause", () => {
  if (segmentEndTime !== null && elements.audioPlayer.currentTime < segmentEndTime) {
    clearSegmentPlayback();
  }
});

// Detection table: click row to play segment
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

// Detection table: keyboard accessibility
elements.detectionsTable.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;

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

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

// Set initial UI state
updateRunButton();
setStatus("Upload audio to begin.");
setModelStatus("Model not loaded", false);
