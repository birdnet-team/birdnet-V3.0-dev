/**
 * Shared type definitions for the BirdNET+ web demo.
 * These types are used by both the main thread and the inference worker.
 */

// -----------------------------------------------------------------------------
// Detection Types
// -----------------------------------------------------------------------------

/**
 * Represents a single bird species detection result.
 * Each detection corresponds to a time segment where a species was identified
 * with confidence above the minimum threshold.
 */
export interface DetectionRow {
  /** Start time of the detection segment in seconds */
  start: number;
  /** End time of the detection segment in seconds */
  end: number;
  /** Scientific name of the detected species (e.g., "Turdus merula") */
  scientific: string;
  /** Common name of the detected species (e.g., "Common Blackbird") */
  common: string;
  /** Model confidence score between 0 and 1 */
  confidence: number;
}

// -----------------------------------------------------------------------------
// Worker Message Types
// -----------------------------------------------------------------------------

/**
 * Message sent to worker to load the ONNX model and species labels.
 * Either provide URLs for lazy loading, or buffers/text for pre-fetched data.
 */
export interface MessageLoadModel {
  type: "loadModel";
  /** Unique request ID for matching responses */
  id: number;
  /** URL to fetch the ONNX model from (if not providing buffer) */
  modelUrl?: string;
  /** Pre-fetched model as ArrayBuffer (if not providing URL) */
  modelBuffer?: ArrayBuffer;
  /** URL to fetch the labels CSV from (if not providing text) */
  labelsUrl?: string;
  /** Pre-fetched labels CSV content (if not providing URL) */
  labelsText?: string;
}

/**
 * Message sent to worker to run inference on audio data.
 */
export interface MessageRunInference {
  type: "runInference";
  /** Unique request ID for matching responses */
  id: number;
  /** Audio samples as Float32Array (mono, already resampled) */
  audioData: Float32Array;
  /** Length of each analysis chunk in seconds (default: 3) */
  chunkLength: number;
  /** Overlap between consecutive chunks in seconds */
  overlap: number;
  /** Number of chunks to process in parallel (batch size) */
  batchSize: number;
  /** Minimum confidence threshold - detections below this are filtered out */
  minConf: number;
  /** Sample rate of the audio data in Hz */
  sampleRate: number;
}

/** Union type for all messages that can be sent to the inference worker */
export type WorkerMessage = MessageLoadModel | MessageRunInference;

// -----------------------------------------------------------------------------
// Worker Response Types
// -----------------------------------------------------------------------------

/**
 * Response from worker after attempting to load the model.
 */
export interface WorkerLoadModelResult {
  type: "loadModelResult";
  id: number;
  success: boolean;
  /** Which execution provider was used: "webgl" (GPU) or "wasm" (CPU) */
  provider?: "webgl" | "wasm";
  /** Number of species labels loaded */
  labelCount?: number;
  /** Error message if loading failed */
  error?: string;
}

/**
 * Response from worker with inference progress updates.
 * Sent periodically during inference so UI can show progress.
 */
export interface WorkerInferenceProgress {
  type: "inferenceProgress";
  id: number;
  /** Number of chunks processed so far */
  current: number;
  /** Total number of chunks to process */
  total: number;
}

/**
 * Response from worker with final inference results.
 */
export interface WorkerInferenceResult {
  type: "inferenceResult";
  id: number;
  success: boolean;
  /** Array of detected species (empty if none found above threshold) */
  detections?: DetectionRow[];
  /** Total inference time in seconds */
  duration?: number;
  /** Error message if inference failed */
  error?: string;
}

/** Union type for all messages that can be received from the inference worker */
export type WorkerResponse =
  | WorkerLoadModelResult
  | WorkerInferenceProgress
  | WorkerInferenceResult;
