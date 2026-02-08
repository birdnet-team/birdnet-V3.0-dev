/**
 * Spectrogram visualization for the BirdNET+ web demo.
 * Computes and renders linear-frequency spectrograms using FFT.js.
 */

import FFT from "fft.js";
import { N_FFT, HOP, DB_RANGE, PRE_EMPHASIS, SAMPLE_RATE } from "./constants";

// -----------------------------------------------------------------------------
// Viridis Colormap
// -----------------------------------------------------------------------------

/**
 * Viridis colormap lookup table (256 RGB entries).
 * A perceptually uniform colormap that's colorblind-friendly.
 * Maps values 0-255 to colors from dark purple/blue to yellow.
 */
const VIRIDIS_COLORS: [number, number, number][] = [
  [68, 1, 84], [68, 2, 86], [69, 4, 87], [69, 5, 89], [70, 7, 90],
  [70, 8, 92], [70, 10, 93], [70, 11, 94], [71, 13, 96], [71, 14, 97],
  [71, 16, 99], [71, 17, 100], [71, 19, 101], [72, 20, 103], [72, 22, 104],
  [72, 23, 105], [72, 24, 106], [72, 26, 108], [72, 27, 109], [72, 28, 110],
  [72, 29, 111], [72, 31, 112], [72, 32, 113], [72, 33, 115], [72, 35, 116],
  [72, 36, 117], [72, 37, 118], [72, 38, 119], [72, 40, 120], [72, 41, 121],
  [71, 42, 122], [71, 44, 122], [71, 45, 123], [71, 46, 124], [71, 47, 125],
  [70, 48, 126], [70, 50, 126], [70, 51, 127], [69, 52, 128], [69, 53, 129],
  [69, 55, 129], [68, 56, 130], [68, 57, 131], [68, 58, 131], [67, 60, 132],
  [67, 61, 132], [66, 62, 133], [66, 63, 133], [66, 64, 134], [65, 66, 134],
  [65, 67, 135], [64, 68, 135], [64, 69, 136], [63, 71, 136], [63, 72, 137],
  [62, 73, 137], [62, 74, 137], [62, 76, 138], [61, 77, 138], [61, 78, 138],
  [60, 79, 139], [60, 80, 139], [59, 82, 139], [59, 83, 140], [58, 84, 140],
  [58, 85, 140], [57, 86, 141], [57, 88, 141], [56, 89, 141], [56, 90, 141],
  [55, 91, 142], [55, 92, 142], [54, 94, 142], [54, 95, 142], [53, 96, 142],
  [53, 97, 143], [52, 98, 143], [52, 99, 143], [51, 101, 143], [51, 102, 143],
  [50, 103, 144], [50, 104, 144], [49, 105, 144], [49, 106, 144], [49, 108, 144],
  [48, 109, 144], [48, 110, 144], [47, 111, 145], [47, 112, 145], [46, 113, 145],
  [46, 114, 145], [45, 116, 145], [45, 117, 145], [44, 118, 145], [44, 119, 145],
  [44, 120, 146], [43, 121, 146], [43, 122, 146], [42, 123, 146], [42, 125, 146],
  [42, 126, 146], [41, 127, 146], [41, 128, 146], [40, 129, 146], [40, 130, 146],
  [40, 131, 146], [39, 132, 146], [39, 133, 146], [38, 134, 146], [38, 136, 146],
  [38, 137, 146], [37, 138, 146], [37, 139, 146], [36, 140, 146], [36, 141, 146],
  [36, 142, 146], [35, 143, 146], [35, 144, 146], [35, 145, 146], [34, 146, 146],
  [34, 147, 146], [33, 148, 146], [33, 149, 146], [33, 150, 146], [32, 151, 145],
  [32, 152, 145], [32, 153, 145], [31, 154, 145], [31, 155, 145], [31, 156, 145],
  [30, 157, 144], [30, 158, 144], [30, 159, 144], [30, 160, 144], [29, 161, 143],
  [29, 162, 143], [29, 163, 143], [29, 164, 142], [28, 165, 142], [28, 166, 142],
  [28, 167, 141], [28, 168, 141], [28, 169, 141], [27, 170, 140], [27, 171, 140],
  [27, 172, 139], [27, 173, 139], [27, 174, 138], [27, 175, 138], [27, 176, 137],
  [27, 177, 137], [27, 178, 136], [27, 179, 136], [27, 180, 135], [27, 181, 135],
  [27, 182, 134], [27, 183, 133], [28, 184, 133], [28, 185, 132], [28, 186, 131],
  [29, 187, 131], [29, 188, 130], [29, 189, 129], [30, 190, 129], [30, 190, 128],
  [31, 191, 127], [31, 192, 126], [32, 193, 126], [33, 194, 125], [33, 195, 124],
  [34, 196, 123], [35, 197, 123], [36, 198, 122], [37, 198, 121], [37, 199, 120],
  [38, 200, 119], [39, 201, 118], [40, 202, 118], [41, 203, 117], [42, 203, 116],
  [44, 204, 115], [45, 205, 114], [46, 206, 113], [47, 207, 112], [49, 207, 111],
  [50, 208, 110], [51, 209, 109], [53, 210, 108], [54, 210, 107], [56, 211, 106],
  [57, 212, 105], [59, 213, 104], [60, 213, 103], [62, 214, 102], [64, 215, 101],
  [65, 215, 100], [67, 216, 98], [69, 217, 97], [70, 217, 96], [72, 218, 95],
  [74, 219, 94], [76, 219, 93], [78, 220, 91], [80, 221, 90], [82, 221, 89],
  [83, 222, 88], [85, 222, 86], [87, 223, 85], [89, 224, 84], [91, 224, 83],
  [94, 225, 81], [96, 225, 80], [98, 226, 79], [100, 226, 77], [102, 227, 76],
  [104, 227, 75], [106, 228, 73], [109, 228, 72], [111, 229, 71], [113, 229, 69],
  [115, 230, 68], [118, 230, 66], [120, 231, 65], [122, 231, 64], [125, 232, 62],
  [127, 232, 61], [129, 232, 59], [132, 233, 58], [134, 233, 56], [137, 234, 55],
  [139, 234, 53], [141, 235, 52], [144, 235, 50], [146, 235, 49], [149, 236, 47],
  [151, 236, 46], [154, 236, 45], [156, 237, 43], [159, 237, 42], [161, 237, 40],
  [163, 238, 39], [166, 238, 38], [168, 238, 36], [171, 239, 35], [173, 239, 34],
  [176, 239, 32], [178, 239, 31], [181, 240, 30], [183, 240, 29], [186, 240, 28],
  [188, 240, 27], [191, 241, 26], [193, 241, 25], [195, 241, 24], [198, 241, 23],
  [200, 241, 23], [203, 241, 22], [205, 242, 22], [207, 242, 21], [210, 242, 21],
  [212, 242, 21], [214, 242, 21], [217, 242, 20], [219, 242, 20], [221, 243, 20],
  [224, 243, 21], [226, 243, 21], [228, 243, 21], [230, 243, 22], [232, 243, 22],
  [235, 243, 23], [237, 244, 24], [239, 244, 25], [241, 244, 26], [243, 244, 27],
  [245, 244, 28], [247, 244, 30], [249, 244, 31], [251, 245, 33], [253, 245, 35]
];

/**
 * Maps a normalized value (0-1) to an RGB color using the viridis colormap.
 *
 * @param value - Normalized value between 0 and 1
 * @returns RGB color tuple [r, g, b]
 */
export function viridisColor(value: number): [number, number, number] {
  // Clamp to [0, 1] range
  const clamped = Math.max(0, Math.min(1, value));
  // Map to colormap index
  const index = Math.floor(clamped * 255);
  return VIRIDIS_COLORS[index];
}

// -----------------------------------------------------------------------------
// Spectrogram Computation
// -----------------------------------------------------------------------------

/**
 * Pre-computed Hann window coefficients for smoother FFT edges.
 * Using a Hann (raised cosine) window reduces spectral leakage.
 */
const hannWindowCache: Map<number, Float32Array> = new Map();

/**
 * Gets or creates a Hann window of the specified size.
 *
 * @param size - Window size in samples
 * @returns Hann window coefficients
 */
function getHannWindow(size: number): Float32Array {
  let window = hannWindowCache.get(size);
  if (!window) {
    window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
    }
    hannWindowCache.set(size, window);
  }
  return window;
}

/**
 * Computes a linear-frequency spectrogram from audio samples.
 * Uses Short-Time Fourier Transform (STFT) with Hann windowing.
 *
 * @param samples - Mono audio samples as Float32Array
 * @param nfft - FFT size (default: 2048)
 * @param hop - Hop size between frames (default: 512)
 * @param preEmphasis - Pre-emphasis coefficient (default: 0.97, 0 to disable)
 * @returns 2D array where spectrogram[time][freq] contains magnitude values
 */
export function computeLinearSpectrogram(
  samples: Float32Array,
  nfft: number = N_FFT,
  hop: number = HOP,
  preEmphasis: number = PRE_EMPHASIS
): Float32Array[] {
  // Apply high-frequency pre-emphasis filter to boost bird calls
  // y[n] = x[n] - α * x[n-1]
  let audio = samples;
  if (preEmphasis > 0) {
    audio = new Float32Array(samples.length);
    audio[0] = samples[0];
    for (let i = 1; i < samples.length; i++) {
      audio[i] = samples[i] - preEmphasis * samples[i - 1];
    }
  }

  // Calculate number of time frames
  const numFrames = Math.max(1, Math.floor((audio.length - nfft) / hop) + 1);
  const freqBins = nfft / 2 + 1;

  // Initialize FFT
  const fft = new FFT(nfft);
  const hannWindow = getHannWindow(nfft);

  // Output array: spectrogram[timeIndex][freqIndex]
  const spectrogram: Float32Array[] = new Array(numFrames);

  // Input/output buffers for FFT
  const fftInput = new Float32Array(nfft);
  const fftOutput = fft.createComplexArray();

  // Process each time frame
  for (let frame = 0; frame < numFrames; frame++) {
    const startSample = frame * hop;

    // Apply Hann window to input frame
    for (let i = 0; i < nfft; i++) {
      const sampleIndex = startSample + i;
      const sample = sampleIndex < audio.length ? audio[sampleIndex] : 0;
      fftInput[i] = sample * hannWindow[i];
    }

    // Compute FFT
    fft.realTransform(fftOutput, fftInput);

    // Extract magnitudes (only positive frequencies, up to Nyquist)
    const magnitudes = new Float32Array(freqBins);
    for (let i = 0; i < freqBins; i++) {
      const real = fftOutput[i * 2];
      const imag = fftOutput[i * 2 + 1];
      // Magnitude = sqrt(real² + imag²)
      magnitudes[i] = Math.sqrt(real * real + imag * imag);
    }

    spectrogram[frame] = magnitudes;
  }

  return spectrogram;
}

// -----------------------------------------------------------------------------
// Spectrogram Rendering
// -----------------------------------------------------------------------------

/**
 * Renders a spectrogram to a canvas element.
 * Uses log-amplitude (dB) scaling and viridis colormap.
 *
 * @param canvas - Target canvas element
 * @param samples - Audio samples to visualize
 * @param nfft - FFT size (default: 2048)
 * @param hop - Hop size (default: 512)
 */
export function renderSpectrogram(
  canvas: HTMLCanvasElement,
  samples: Float32Array,
  nfft: number = N_FFT,
  hop: number = HOP
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Compute spectrogram
  const spectrogram = computeLinearSpectrogram(samples, nfft, hop);
  const numFrames = spectrogram.length;
  const freqBins = spectrogram[0]?.length || 0;

  if (numFrames === 0 || freqBins === 0) {
    // No data - clear canvas
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  // Find global max for normalization
  let globalMax = 0;
  for (let t = 0; t < numFrames; t++) {
    for (let f = 0; f < freqBins; f++) {
      if (spectrogram[t][f] > globalMax) {
        globalMax = spectrogram[t][f];
      }
    }
  }

  // Avoid log(0) issues
  const minValue = globalMax * 1e-10;
  const maxDb = 20 * Math.log10(Math.max(globalMax, minValue));

  // Create image data for pixel manipulation
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  // Render each pixel
  for (let y = 0; y < height; y++) {
    // Map canvas Y (top=high freq) to frequency bin
    const freqIndex = Math.floor(((height - 1 - y) / height) * freqBins);

    for (let x = 0; x < width; x++) {
      // Map canvas X to time frame
      const timeIndex = Math.floor((x / width) * numFrames);

      // Get magnitude and convert to dB
      const magnitude = spectrogram[timeIndex]?.[freqIndex] ?? 0;
      const db = 20 * Math.log10(Math.max(magnitude, minValue));

      // Normalize within dynamic range
      const normalized = Math.max(0, Math.min(1, (db - (maxDb - DB_RANGE)) / DB_RANGE));

      // Apply viridis colormap
      const [r, g, b] = viridisColor(normalized);

      // Write pixel (RGBA format, 4 bytes per pixel)
      const pixelIndex = (y * width + x) * 4;
      pixels[pixelIndex] = r;
      pixels[pixelIndex + 1] = g;
      pixels[pixelIndex + 2] = b;
      pixels[pixelIndex + 3] = 255; // Fully opaque
    }
  }

  // Draw to canvas
  ctx.putImageData(imageData, 0, 0);
}

/**
 * Gets the frequency at a given Y position in the spectrogram.
 * Useful for displaying frequency values on hover.
 *
 * @param y - Y coordinate (0 = top of canvas)
 * @param canvasHeight - Total canvas height
 * @param sampleRate - Audio sample rate (default: 32000)
 * @param nfft - FFT size (default: 2048)
 * @returns Frequency in Hz
 */
export function getFrequencyAtY(
  y: number,
  canvasHeight: number,
  sampleRate: number = SAMPLE_RATE,
  nfft: number = N_FFT
): number {
  // Y=0 is top (Nyquist), Y=height is bottom (0 Hz)
  const normalizedY = (canvasHeight - y) / canvasHeight;
  const nyquist = sampleRate / 2;
  return normalizedY * nyquist;
}

/**
 * Gets the time at a given X position in the spectrogram.
 *
 * @param x - X coordinate
 * @param canvasWidth - Total canvas width
 * @param audioDuration - Total audio duration in seconds
 * @returns Time in seconds
 */
export function getTimeAtX(
  x: number,
  canvasWidth: number,
  audioDuration: number
): number {
  return (x / canvasWidth) * audioDuration;
}
