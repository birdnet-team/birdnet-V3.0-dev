import * as ort from "onnxruntime-web";

// Configure WASM paths for worker context
ort.env.wasm.wasmPaths = "/ort/";

let session: ort.InferenceSession | null = null;
let labelScientific: string[] = [];
let labelCommon: string[] = [];

type MessageLoadModel = {
  type: "loadModel";
  id: number;
  modelUrl?: string;
  modelBuffer?: ArrayBuffer;
  labelsUrl?: string;
  labelsText?: string;
};

type MessageRunInference = {
  type: "runInference";
  id: number;
  audioData: Float32Array;
  chunkLength: number;
  overlap: number;
  batchSize: number;
  minConf: number;
  sampleRate: number;
};

type WorkerMessage = MessageLoadModel | MessageRunInference;

type DetectionRow = {
  start: number;
  end: number;
  scientific: string;
  common: string;
  confidence: number;
};

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

function parseLabelsCsv(text: string) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0) return { scientific: [], common: [] };
  const header = parseCsvLine(lines[0], ";");
  const sciIndex = header.indexOf("sci_name");
  const comIndex = header.indexOf("com_name");
  const scientific: string[] = [];
  const common: string[] = [];
  for (let i = 1; i < lines.length; i += 1) {
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

async function loadLabelsFromUrl(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch labels: ${res.status}`);
  }
  const text = await res.text();
  return parseLabelsCsv(text);
}

async function loadModelSession(source: string | ArrayBuffer) {
  // Try WebGL first, fall back to WASM
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const createSession = (s: any, providers: string[]) =>
    ort.InferenceSession.create(s, {
      executionProviders: providers,
      graphOptimizationLevel: "all"
    });

  try {
    const webglSession = await createSession(source, ["webgl"]);
    return { session: webglSession, provider: "webgl" as const };
  } catch {
    const wasmSession = await createSession(source, ["wasm"]);
    return { session: wasmSession, provider: "wasm" as const };
  }
}

function chunkAudioPlan(
  audioData: Float32Array,
  chunkLengthSec: number,
  overlapSec: number,
  sampleRate: number
) {
  const chunkSamples = Math.floor(chunkLengthSec * sampleRate);
  const hopSamples = Math.max(1, Math.floor((chunkLengthSec - overlapSec) * sampleRate));
  const starts: number[] = [];
  const spans: [number, number][] = [];
  for (let i = 0; i + chunkSamples <= audioData.length; i += hopSamples) {
    starts.push(i);
    spans.push([i / sampleRate, (i + chunkSamples) / sampleRate]);
  }
  return { starts, spans, chunkSamples };
}

function pickPredictionTensor(outputs: Record<string, ort.Tensor>, labelCount: number) {
  const tensors = Object.values(outputs);
  const byLabelCount = tensors.find((t) => t.dims?.length === 2 && t.dims[1] === labelCount);
  if (byLabelCount) return byLabelCount;
  return tensors[0];
}

async function handleLoadModel(msg: MessageLoadModel) {
  try {
    // Load labels
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
      throw new Error("No labels parsed from CSV.");
    }

    // Load model
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

    self.postMessage({
      type: "loadModelResult",
      id: msg.id,
      success: true,
      provider: result.provider,
      labelCount: labelScientific.length
    });
  } catch (err) {
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

async function handleRunInference(msg: MessageRunInference) {
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
    const { starts, spans, chunkSamples } = chunkAudioPlan(audioData, chunkLength, overlap, sampleRate);

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
    const audioLen = audioData.length;
    const detections: DetectionRow[] = [];
    const startTime = performance.now();

    for (let i = 0; i < starts.length; i += batchSize) {
      const batchCount = Math.min(batchSize, starts.length - i);
      const input = new Float32Array(batchCount * chunkSamples);

      for (let b = 0; b < batchCount; b += 1) {
        const startSample = starts[i + b];
        const endSample = Math.min(startSample + chunkSamples, audioLen);
        input.set(audioData.subarray(startSample, endSample), b * chunkSamples);
      }

      const tensor = new ort.Tensor("float32", input, [batchCount, chunkSamples]);
      const feeds: Record<string, ort.Tensor> = {
        [session.inputNames[0]]: tensor
      };

      // Report progress
      self.postMessage({
        type: "inferenceProgress",
        id: msg.id,
        current: Math.min(i + batchCount, starts.length),
        total: starts.length
      });

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
            detections.push({
              start,
              end,
              scientific: labelScientific[c] || "",
              common: labelCommon[c] || "",
              confidence: val
            });
          }
        }
      }
    }

    const duration = (performance.now() - startTime) / 1000;
    detections.sort((a, b) => (a.start === b.start ? b.confidence - a.confidence : a.start - b.start));

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

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const msg = event.data;
  switch (msg.type) {
    case "loadModel":
      await handleLoadModel(msg);
      break;
    case "runInference":
      await handleRunInference(msg);
      break;
  }
};
