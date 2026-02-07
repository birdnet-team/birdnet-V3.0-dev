/**
 * Application constants for the BirdNET+ web demo.
 * Centralized configuration for audio processing, model paths, and defaults.
 */

// -----------------------------------------------------------------------------
// Model Configuration
// -----------------------------------------------------------------------------

/**
 * Default URL for the BirdNET+ ONNX model.
 * Using the V3.0-preview3 Global model with 11K species support.
 */
export const MODEL_URL =
  "./models/BirdNET+_V3.0-preview3_Global_11K_FP32.onnx";

/**
 * Default URL for the species labels CSV file.
 * Contains scientific and common names for all 11K supported species.
 */
export const LABELS_URL =
  "./models/BirdNET+_V3.0-preview3_Global_11K_Labels.csv";

// -----------------------------------------------------------------------------
// Audio Processing Constants
// -----------------------------------------------------------------------------

/**
 * Target sample rate in Hz. BirdNET models expect 32kHz mono audio.
 * All input audio is resampled to this rate before inference.
 */
export const SAMPLE_RATE = 32000;

/**
 * FFT size for spectrogram computation.
 * 2048 samples at 32kHz gives ~64ms window, good frequency resolution for birdsong.
 */
export const N_FFT = 2048;

/**
 * Hop size (stride) between consecutive FFT windows.
 * 512 samples = 75% overlap, providing smooth temporal resolution.
 */
export const HOP = 512;

// -----------------------------------------------------------------------------
// Inference Defaults
// -----------------------------------------------------------------------------

/** Default analysis chunk length in seconds */
export const DEFAULT_CHUNK_LENGTH = 3;

/** Default overlap between chunks in seconds */
export const DEFAULT_OVERLAP = 0;

/** Default batch size for inference */
export const DEFAULT_BATCH_SIZE = 1;

/** Default minimum confidence threshold */
export const DEFAULT_MIN_CONF = 0.25;

// -----------------------------------------------------------------------------
// Spectrogram Visualization
// -----------------------------------------------------------------------------

/**
 * Dynamic range in decibels for spectrogram visualization.
 * Values below (max - DB_RANGE) are clipped to black.
 */
export const DB_RANGE = 80;

/**
 * Pre-emphasis coefficient for high-frequency boost.
 * Helps make bird calls more visible in spectrogram.
 * Set to 0 to disable pre-emphasis filtering.
 */
export const PRE_EMPHASIS = 0.97;
