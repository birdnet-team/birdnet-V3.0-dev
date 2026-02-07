/**
 * Audio processing utilities for the BirdNET+ web demo.
 * Handles decoding, resampling, and channel mixing of audio files.
 */

import { SAMPLE_RATE } from "./constants";

// -----------------------------------------------------------------------------
// Audio Decoding
// -----------------------------------------------------------------------------

/**
 * Decodes an audio file into a mono Float32Array at the target sample rate.
 * Supports any format the browser's AudioContext can decode (MP3, WAV, OGG, etc.)
 *
 * @param audioFile - The audio file to decode (from file input or drop)
 * @param targetSampleRate - Target sample rate (default: 32000 Hz for BirdNET)
 * @returns Promise resolving to mono audio samples as Float32Array
 * @throws Error if decoding fails or file is empty
 */
export async function decodeAudioFile(
  audioFile: File,
  targetSampleRate: number = SAMPLE_RATE
): Promise<Float32Array> {
  // Read file as ArrayBuffer
  const arrayBuffer = await audioFile.arrayBuffer();

  // Use OfflineAudioContext to decode and resample in one step
  // This is more accurate than manual resampling for most cases
  const audioCtx = new OfflineAudioContext(
    1, // mono output
    1, // temporary length, will be replaced
    targetSampleRate
  );

  // Decode the compressed audio
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // Convert to mono and resample
  return extractMonoChannel(audioBuffer, targetSampleRate);
}

/**
 * Extracts a mono channel from an AudioBuffer and resamples if needed.
 * Multi-channel audio is mixed down to mono by averaging all channels.
 *
 * @param audioBuffer - Decoded AudioBuffer from Web Audio API
 * @param targetSampleRate - Target sample rate
 * @returns Mono audio samples as Float32Array
 */
function extractMonoChannel(
  audioBuffer: AudioBuffer,
  targetSampleRate: number
): Float32Array {
  const numChannels = audioBuffer.numberOfChannels;
  const sourceSampleRate = audioBuffer.sampleRate;

  // Get mixed mono samples at source sample rate
  let mono: Float32Array;
  if (numChannels === 1) {
    mono = audioBuffer.getChannelData(0);
  } else {
    // Mix all channels to mono by averaging
    mono = mixToMono(audioBuffer);
  }

  // Resample if rates don't match
  if (sourceSampleRate !== targetSampleRate) {
    return resampleLinear(mono, sourceSampleRate, targetSampleRate);
  }

  // Return a copy to avoid holding reference to AudioBuffer
  return new Float32Array(mono);
}

/**
 * Mixes multi-channel audio down to mono by averaging all channels.
 *
 * @param audioBuffer - Multi-channel AudioBuffer
 * @returns Mono samples as Float32Array
 */
export function mixToMono(audioBuffer: AudioBuffer): Float32Array {
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const mono = new Float32Array(length);

  // Sum all channels
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i];
    }
  }

  // Divide by channel count to get average
  const scale = 1 / numChannels;
  for (let i = 0; i < length; i++) {
    mono[i] *= scale;
  }

  return mono;
}

// -----------------------------------------------------------------------------
// Resampling
// -----------------------------------------------------------------------------

/**
 * Resamples audio using linear interpolation.
 * Simple but effective for the sample rate conversions we need.
 *
 * For higher quality resampling, consider using the OfflineAudioContext
 * which uses sinc interpolation internally.
 *
 * @param input - Input audio samples
 * @param srcRate - Source sample rate in Hz
 * @param dstRate - Destination sample rate in Hz
 * @returns Resampled audio as Float32Array
 */
export function resampleLinear(
  input: Float32Array,
  srcRate: number,
  dstRate: number
): Float32Array {
  // Calculate output length based on ratio
  const ratio = srcRate / dstRate;
  const outLen = Math.floor(input.length / ratio);
  const output = new Float32Array(outLen);

  // Linear interpolation between samples
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const srcIndex = Math.floor(srcPos);
    const frac = srcPos - srcIndex;

    // Get surrounding samples (clamp to valid range)
    const s0 = input[srcIndex] || 0;
    const s1 = input[srcIndex + 1] || s0;

    // Interpolate
    output[i] = s0 + frac * (s1 - s0);
  }

  return output;
}

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

/**
 * Formats a time value in seconds to MM:SS or HH:MM:SS format.
 *
 * @param seconds - Time in seconds
 * @returns Formatted time string
 */
export function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);

  if (h > 0) {
    return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  }
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Gets the total duration of audio data in seconds.
 *
 * @param samples - Audio samples as Float32Array
 * @param sampleRate - Sample rate in Hz (default: 32000)
 * @returns Duration in seconds
 */
export function getAudioDuration(
  samples: Float32Array,
  sampleRate: number = SAMPLE_RATE
): number {
  return samples.length / sampleRate;
}
