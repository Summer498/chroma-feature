// app.ts – chroma vector visualizer with optional smoothing & softmax normalization
/*****************************************************************************************
 * IMPORTS
 *****************************************************************************************/
import Meyda from "meyda";

type MeydaAnalyzer = ReturnType<typeof Meyda.createMeydaAnalyzer>;

/*****************************************************************************************
 * UI ELEMENTS & CANVAS SETUP
 *****************************************************************************************/
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;
const canvas = document.getElementById("chromaCanvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

function resizeCanvas(): void {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

/*****************************************************************************************
 * CONSTANTS & SETTINGS
 *****************************************************************************************/
const FFT_SIZE = 4096 as const;
const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

const PITCH_ORDERS = {
  semitone: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  fifths: [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5],
} as const;

const meydaToggle = document.getElementById("meydaToggle") as HTMLInputElement;
const softmaxToggle = document.getElementById("softmaxToggle") as HTMLInputElement;
const fifthToggle = document.getElementById("fifthToggle") as HTMLInputElement;
const smoothSlider = document.getElementById("smoothSlider") as HTMLInputElement;

/*****************************************************************************************
 * UTILITY FUNCTIONS
 *****************************************************************************************/
function getSmoothRate(): number {
  return parseFloat(smoothSlider.value);
}
function isSoftmax(): boolean {
  return softmaxToggle.checked;
}
function orderMode(): keyof typeof PITCH_ORDERS {
  return fifthToggle.checked ? "fifths" : "semitone";
}
function useMeyda(): boolean {
  return meydaToggle.checked;
}

/*****************************************************************************************
 * GLOBAL STATE
 *****************************************************************************************/
let audioContext: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let stream: MediaStream | null = null;
let animationId: number | null = null;
let meydaAnalyzer: MeydaAnalyzer | null = null;
let lastChroma: Float32Array | null = null;

/*****************************************************************************************
 * UTILITY FUNCTIONS
 *****************************************************************************************/
function freqToPitchClass(freq: number): number | null {
  if (freq < 50 || freq > 5000) return null;
  const midi = Math.round(12 * Math.log2(freq / 440) + 69);
  return ((midi % 12) + 12) % 12;
}

function accumulateChroma(freqBuffer: Uint8Array, sampleRate: number): Float32Array {
  const chroma = new Float32Array(12);
  const binWidth = sampleRate / (freqBuffer.length * 2);
  let total = 0;
  for (let i = 0; i < freqBuffer.length; i++) {
    const amp = freqBuffer[i] / 255;
    if (amp === 0) continue;
    const freq = i * binWidth;
    const pc = freqToPitchClass(freq);
    if (pc === null) continue;
    chroma[pc] += amp*amp;
    total += amp*amp;
  }
  // --- normalize so that sum == 1, keeping each value in 0‑1 ---
  if (total > 0) {
    for (let i = 0; i < 12; i++) chroma[i] /= total;
  }
  return chroma.map(e=>Math.sqrt(e));
}

function softmax(vec: Float32Array): Float32Array {
  const out = new Float32Array(12);
  let max = -Infinity;
  for (const v of vec) max = Math.max(max, v);
  let sum = 0;
  for (let i = 0; i < 12; i++) {
    out[i] = Math.exp(vec[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < 12; i++) out[i] /= sum;
  return out;
}

function smoothChroma(src: Float32Array): Float32Array {
  if (lastChroma === null) {
    lastChroma = src.slice();
    return lastChroma;
  }
  for (let i = 0; i < 12; i++) {
    lastChroma[i] = getSmoothRate() * lastChroma[i] + (1 - getSmoothRate()) * src[i];
  }
  return lastChroma;
}

function preprocessChroma(raw: Float32Array): Float32Array {
  const withNorm = isSoftmax() ? softmax(raw) : raw;
  return getSmoothRate() > 0 ? smoothChroma(withNorm) : withNorm;
}

/*****************************************************************************************
 * DRAWING
 *****************************************************************************************/
function drawChroma(rawChroma: Float32Array): void {
  const chroma = preprocessChroma(rawChroma);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const barH = canvas.height / 12;
  const order = PITCH_ORDERS[orderMode()];
  for (let i = 0; i < 12; i++) {
    const idx = order[i];
    const val = chroma[idx];
    const barW = val * canvas.width;
    const y = canvas.height - (i + 1) * barH;
    ctx.fillStyle = `hsl(${(idx * 150 + 120) % 360} 70% 50%)`;
    ctx.fillRect(0, y, barW, barH * 0.9);
    ctx.fillStyle = "#ffffffb0";
    ctx.font = `${barH * 0.5}px sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillText(PITCH_NAMES[idx], 8, y + barH / 2);
  }
}

/*****************************************************************************************
 * AUDIO CAPTURE & ANALYSIS SETUP
 *****************************************************************************************/
interface CaptureBundle {
  stream: MediaStream;
  audioContext: AudioContext;
  analyser: AnalyserNode;
  source: MediaStreamAudioSourceNode;
}

async function createDisplayAudioAnalyser(): Promise<CaptureBundle> {
  const stream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  source.connect(analyser);
  return { stream, audioContext, analyser, source };
}

function createMeydaChroma(audioCtx: AudioContext, src: MediaStreamAudioSourceNode, cb: (d: Float32Array)=>void): MeydaAnalyzer {
  return Meyda.createMeydaAnalyzer({
    audioContext: audioCtx,
    source: src,
    bufferSize: 4096,
    hopSize: 2048,
    featureExtractors: ["chroma"],
    callback: (f: { chroma?: number[] }) => { if (f.chroma) cb(Float32Array.from(f.chroma)); },
  });
}

/*****************************************************************************************
 * START / STOP
 *****************************************************************************************/
async function startCapture(): Promise<void> {
  try {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    lastChroma = null;

    const bundle = await createDisplayAudioAnalyser();
    ({ stream, audioContext, analyser } = bundle);

    if (useMeyda()) {
      meydaAnalyzer = createMeydaChroma(bundle.audioContext, bundle.source, drawChroma);
      meydaAnalyzer.start();
    } else {
      const freqBuffer = new Uint8Array(analyser.frequencyBinCount);
      const loop = (): void => {
        if (!analyser) return;
        analyser.getByteFrequencyData(freqBuffer);
        drawChroma(accumulateChroma(freqBuffer, audioContext!.sampleRate));
        animationId = requestAnimationFrame(loop);
      };
      loop();
    }
  } catch (e) {
    alert(`Error: ${(e as Error).message}`);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function stopCapture(): void {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  if (animationId !== null) cancelAnimationFrame(animationId);
  if (meydaAnalyzer) { meydaAnalyzer.stop(); meydaAnalyzer = null; }
  if (analyser) { analyser.disconnect(); analyser = null; }
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  ctx.clearRect(0,0,canvas.width,canvas.height);
}

startBtn.addEventListener("click", () => void startCapture());
stopBtn.addEventListener("click", stopCapture);
