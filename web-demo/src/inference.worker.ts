/**
 * BirdNET+ Inference Web Worker
 *
 * Runs ONNX model inference in a separate thread to prevent UI freezing.
 * Communicates with main thread via postMessage API.
 *
 * Responsibilities:
 * - Loading the ONNX model (with WebGL preference, WASM fallback)
 * - Parsing species labels from CSV
 * - Chunking audio and running batched inference
 * - Reporting progress during long inference runs
 */

import * as ort from "onnxruntime-web";
import type {
  DetectionRow,
  MessageLoadModel,
  MessageRunInference,
  WorkerMessage
} from "./types";

// -----------------------------------------------------------------------------
// ONNX Runtime Configuration
// -----------------------------------------------------------------------------

// Configure WASM paths for worker context
// Files are served from /ort/ in the public folder
ort.env.wasm.wasmPaths = "/ort/";

// -----------------------------------------------------------------------------
// Worker State
// -----------------------------------------------------------------------------

/** Loaded ONNX inference session */
let session: ort.InferenceSession | null = null;

/** Scientific names indexed by model output position */
let labelScientific: string[] = [];

/** Common names indexed by model output position */
let labelCommon: string[] = [];

// -----------------------------------------------------------------------------
// CSV Parsing
// -----------------------------------------------------------------------------

/**
 * Parses a single CSV line, handling quoted fields.
 * Supports fields with commas inside quotes.
 *
 * @param line - Single line from CSV file
 * @param delimiter - Field separator (semicolon for BirdNET labels)
 * @returns Array of field values
 */
function parseCsvLine(line: string, delimiter: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === delimiter && !inQuotes) {
      fields.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  fields.push(current);
  return fields;
}

/**
 * Parses the BirdNET labels CSV file.
 * Extracts scientific and common names for each species.
 *
 * @param csvText - Raw CSV content
 * @returns Object with scientific and common name arrays
 */
function parseLabelsCsv(csvText: string): {
  scientific: string[];
  common: string[];
} {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length === 0) {
    return { scientific: [], common: [] };
  }

  // Parse header to find column indices
  const header = parseCsvLine(lines[0], ";");
  const sciIndex = header.indexOf("sci_name");
  const comIndex = header.indexOf("com_name");

  const scientific: string[] = [];
  const common: string[] = [];

  // Parse data rows
  for (let i = 1; i < lines.length; i++) {
    const cols = parseCsvLine(lines[i], ";");
    const sci = (cols[sciIndex] || "").trim();
    const com = (cols[comIndex] || "").trim();

    if (sci || com) {
      scientific.push(sci);
      common.push(com);
    }
  }

  return { scientific, common };
}

/**
 * Fetches and parses labels from a URL.
 *
 * @param url - URL to labels CSV file
 * @returns Parsed scientific and common names
 */
async function loadLabelsFromUrl(url: string): Promise<{
  scientific: string[];
  common: string[];
}> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch labels: ${response.status}`);
  }
  const text = await response.text();
  return parseLabelsCsv(text);
}

// -----------------------------------------------------------------------------
// Model Loading
// -----------------------------------------------------------------------------

/**
 * Loads an ONNX inference session with preferred execution provider.
 * Tries WebGL (GPU) first for best performance, falls back to WASM (CPU).
 *
 * @param source - Model URL or ArrayBuffer
 * @returns Session and provider info
 */
async function loadModelSession(
  source: string | ArrayBuffer,
): Promise<{
  session: ort.InferenceSession;
  provider: "webgl" | "wasm";
}> {
  // Convert ArrayBuffer to Uint8Array (ONNX runtime expects Uint8Array for binary data)
  const modelData: string | Uint8Array =
    typeof source === "string" ? source : new Uint8Array(source);

  // Note: ONNX Runtime types are overly strict - the API actually accepts string URLs
  // Using type assertion to work around incomplete type definitions
  const createSession = (data: string | Uint8Array, providers: string[]) =>
    ort.InferenceSession.create(data as Parameters<typeof ort.InferenceSession.create>[0], {
      executionProviders: providers,
    });

  // Try WebGL first (GPU acceleration), fall back to WASM (CPU)
  try {
    const webglSession = await createSession(modelData, ["webgl"]);
    return { session: webglSession, provider: "webgl" };
  } catch {
    const wasmSession = await createSession(modelData, ["wasm"]);
    return { session: wasmSession, provider: "wasm" };
  }
}

// -----------------------------------------------------------------------------
// Audio Chunking
// -----------------------------------------------------------------------------

/**
 * Plans how to split audio into overlapping chunks for inference.
 * BirdNET processes audio in fixed-length segments (default 3s).
 *
 * @param audioLength - Total audio samples
 * @param chunkLengthSec - Chunk duration in seconds
 * @param overlapSec - Overlap between chunks in seconds
 * @param sampleRate - Audio sample rate
 * @returns Chunk start positions and time spans
 */
function planAudioChunks(
  audioLength: number,
  chunkLengthSec: number,
  overlapSec: number,
  sampleRate: number
): {
  starts: number[];
  spans: [number, number][];
  chunkSamples: number;
} {
  const chunkSamples = Math.floor(chunkLengthSec * sampleRate);
  const hopSamples = Math.max(
    1,
    Math.floor((chunkLengthSec - overlapSec) * sampleRate)
  );

  const starts: number[] = [];
  const spans: [number, number][] = [];

  for (let i = 0; i + chunkSamples <= audioLength; i += hopSamples) {
    starts.push(i);
    spans.push([i / sampleRate, (i + chunkSamples) / sampleRate]);
  }

  return { starts, spans, chunkSamples };
}

/**
 * Finds the appropriate output tensor from model results.
 * Different ONNX exports may have different output names.
 *
 * @param outputs - Model output tensors
 * @param labelCount - Expected number of species labels
 * @returns The prediction tensor
 */
function findPredictionTensor(
  outputs: Record<string, ort.Tensor>,
  labelCount: number
): ort.Tensor {
  const tensors = Object.values(outputs);

  // Look for tensor with expected shape [batch, num_species]
  const byLabelCount = tensors.find(
    (t) => t.dims?.length === 2 && t.dims[1] === labelCount
  );

  if (byLabelCount) return byLabelCount;

  // Fall back to first tensor
  return tensors[0];
}

// -----------------------------------------------------------------------------
// Message Handlers
// -----------------------------------------------------------------------------

/**
 * Handles model loading request from main thread.
 * Loads labels and ONNX model, reports success/failure.
 */
async function handleLoadModel(msg: MessageLoadModel): Promise<void> {
  try {
    // Load species labels
    if (msg.labelsText) {
      const parsed = parseLabelsCsv(msg.labelsText);
      labelScientific = parsed.scientific;
      labelCommon = parsed.common;
    } else if (msg.labelsUrl) {
      const parsed = await loadLabelsFromUrl(msg.labelsUrl);
      labelScientific = parsed.scientific;
      labelCommon = parsed.common;
    } else {
      throw new Error("No labels provided");
    }

    if (!labelScientific.length) {
      throw new Error("No labels parsed from CSV");
    }

    // Load ONNX model
    let source: string | ArrayBuffer;
    if (msg.modelBuffer) {
      source = msg.modelBuffer;
    } else if (msg.modelUrl) {
      source = msg.modelUrl;
    } else {
      throw new Error("No model provided");
    }

    const result = await loadModelSession(source);
    session = result.session;

    // Report success
    self.postMessage({
      type: "loadModelResult",
      id: msg.id,
      success: true,
      provider: result.provider,
      labelCount: labelScientific.length
    });
  } catch (err) {
    // Reset state on failure
    session = null;
    labelScientific = [];
    labelCommon = [];

    self.postMessage({
      type: "loadModelResult",
      id: msg.id,
      success: false,
      error: (err as Error).message
    });
  }
}

/**
 * Handles inference request from main thread.
 * Processes audio in batched chunks and streams progress updates.
 */
async function handleRunInference(msg: MessageRunInference): Promise<void> {
  if (!session) {
    self.postMessage({
      type: "inferenceResult",
      id: msg.id,
      success: false,
      error: "Model not loaded"
    });
    return;
  }

  try {
    const { audioData, chunkLength, overlap, batchSize, minConf, sampleRate } = msg;

    // Plan chunk positions
    const { starts, spans, chunkSamples } = planAudioChunks(
      audioData.length,
      chunkLength,
      overlap,
      sampleRate
    );

    // Handle edge case: audio shorter than one chunk
    if (!starts.length) {
      self.postMessage({
        type: "inferenceResult",
        id: msg.id,
        success: true,
        detections: [],
        duration: 0
      });
      return;
    }

    const labelCount = labelScientific.length;
    const detections: DetectionRow[] = [];
    const startTime = performance.now();

    // Process chunks in batches
    for (let i = 0; i < starts.length; i += batchSize) {
      const batchCount = Math.min(batchSize, starts.length - i);

      // Prepare input tensor: [batchCount, chunkSamples]
      const input = new Float32Array(batchCount * chunkSamples);
      for (let b = 0; b < batchCount; b++) {
        const startSample = starts[i + b];
        const endSample = Math.min(startSample + chunkSamples, audioData.length);
        input.set(audioData.subarray(startSample, endSample), b * chunkSamples);
      }

      // Create ONNX tensor and run inference (I/O is always float32)
      const tensor = new ort.Tensor("float32", input, [batchCount, chunkSamples]);
      const feeds: Record<string, ort.Tensor> = {
        [session.inputNames[0]]: tensor
      };

      // Report progress to main thread
      self.postMessage({
        type: "inferenceProgress",
        id: msg.id,
        current: Math.min(i + batchCount, starts.length),
        total: starts.length
      });

      // Run model
      const results = await session.run(feeds);

      const predTensor = findPredictionTensor(results, labelCount);
      const predictions = predTensor.data as Float32Array;
      const outputDim = predTensor.dims?.[1] || labelCount;

      // Extract detections above threshold
      for (let b = 0; b < batchCount; b++) {
        const offset = b * outputDim;
        const [start, end] = spans[i + b];

        for (let c = 0; c < labelCount; c++) {
          const confidence = predictions[offset + c];
          if (confidence >= minConf) {
            detections.push({
              start,
              end,
              scientific: labelScientific[c] || "",
              common: labelCommon[c] || "",
              confidence
            });
          }
        }
      }
    }

    // Calculate total inference time
    const duration = (performance.now() - startTime) / 1000;

    // Sort by time, then by confidence (for same-time detections)
    detections.sort((a, b) =>
      a.start === b.start ? b.confidence - a.confidence : a.start - b.start
    );

    // Report results
    self.postMessage({
      type: "inferenceResult",
      id: msg.id,
      success: true,
      detections,
      duration
    });
  } catch (err) {
    self.postMessage({
      type: "inferenceResult",
      id: msg.id,
      success: false,
      error: (err as Error).message
    });
  }
}

// -----------------------------------------------------------------------------
// Message Router
// -----------------------------------------------------------------------------

/**
 * Main message handler - routes incoming messages to appropriate handler.
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const msg = event.data;

  switch (msg.type) {
    case "loadModel":
      await handleLoadModel(msg);
      break;
    case "runInference":
      await handleRunInference(msg);
      break;
    default:
      console.warn("Unknown worker message type:", (msg as { type: string }).type);
  }
};
