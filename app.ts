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
const rankToggle = document.getElementById("rankToggle") as HTMLInputElement;
const percToggle = document.getElementById('percToggle') as HTMLInputElement;
const squareToggle = document.getElementById('squareToggle') as HTMLInputElement;
const shepardToggle = document.getElementById('shepardToggle') as HTMLInputElement;
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
function useRank(): boolean {
  return rankToggle.checked;
}
function useSquare(): boolean {
  return squareToggle.checked;
}
function useShepard(): boolean {
  return shepardToggle.checked;
}
function usePercFilter(): boolean {
  return percToggle.checked;
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
// Median HPSS
const HISTORY = 25;
const history: Uint8Array[] = [];

function medianFilter(buf: Uint8Array): Uint8Array {
  console.log("median")
  const mags = buf;
  history.push(mags);
  if (history.length > HISTORY) history.shift();

  const out = new Uint8Array(mags.length);
  for (let i = 0; i < mags.length; i++) {
    const column = history.map(h => h[i]).sort((a, b) => a - b);
    out[i] = column[Math.floor(column.length / 2)];
  }
  return out;
}


function freqToPitchClass(freq: number): number | null {
  if (freq < 50 || freq > 2500) return null;
  const midi = Math.round(12 * Math.log2(freq / 440) + 69);
  return ((midi % 12) + 12) % 12;
}

function accumulateChroma(freqBuffer: Uint8Array, sampleRate: number): Float32Array {
  const chroma = new Float32Array(12);
  const binWidth = sampleRate / (freqBuffer.length * 2);
  for (let i = 0; i < freqBuffer.length; i++) {
    const amp = 1.0 * freqBuffer[i] / 255;
    if (amp === 0) continue;
    const freq = i * binWidth;
    const pc = freqToPitchClass(freq);
    if (pc === null) continue;
    chroma[pc] += amp * amp;
  }

  return chroma.map(e => Math.sqrt(e)).map(e => e / 3);
}

function rankVector(v: Float32Array): Float32Array {
  const idx = [...Array(12).keys()].sort((a, b) => v[b] - v[a]);
  const out = new Float32Array(12);
  idx.forEach((p, r) => { out[p] = Math.exp(-r); });
  //  idx.forEach((p, r) => { out[p] = 1 - r / 11; });
  return out;
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
  return out.map(e => e * 5);
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
  return [raw]
    .map(e => useRank() ? rankVector(e) : e)
    .map(e => isSoftmax() ? softmax(e) : e)
    .map(e => getSmoothRate() > 0 ? smoothChroma(e) : e)
  [0];
}

/*-------------------------------------------------------------*
 * 1. いまのフレームで “1 位” のピッチクラスを返す
 *-------------------------------------------------------------*/
function topPitchClass(chroma: Float32Array): number {
  let maxIdx = 0, maxVal = chroma[0];
  for (let i = 1; i < 12; i++) if (chroma[i] > maxVal) {
    maxVal = chroma[i]; maxIdx = i;
  }
  return maxIdx;          // 0=C, 1=C# … 11=B
}

/*───────────────────────────────────────────────*
 * Shepard tone generator – 無限音階
 *───────────────────────────────────────────────*/
class ShepardTone {
  private ctx = new AudioContext();
  private gain = this.ctx.createGain();
  private oscs: OscillatorNode[] = [];
  private readonly octaves = [-3, -2, -1, 0, 1, 2, 3]; // 55Hz〜7kHz 付近
  private readonly center = 440;                 // 振幅中心周波数
  private readonly width = 1.5;                 // ガウス幅 (oct)
  private currentPC = -1;

  constructor() {
    this.gain.gain.value = 0;
    this.gain.connect(this.ctx.destination);
  }

  /** ピッチクラスを指定 (0=C … 11=B, -1=ミュート) */
  play(pc: number): void {
    if (!useShepard()) {
      /* ←トグルが OFF のときは必ず音量 0 に */
      this.gain.gain.setTargetAtTime(0, oscCtx.currentTime, 0.02);
      prevPC = -1;                    // 状態もリセット
      return;
    }
    if (pc === this.currentPC) return;
    this.currentPC = pc;
    /* 全オシレータを止める */
    this.oscs.forEach(o => o.stop());
    this.oscs.length = 0;
    if (pc < 0) {
      this.gain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.05);
      return;
    }
    const f0 = 440 * Math.pow(2, (pc - 9) / 12); // A4=440基準
    /* 新しい Shepard 音を生成 */
    this.octaves.forEach(oct => {
      const f = f0 * Math.pow(2, oct);
      if (f < 20 || f > this.ctx.sampleRate / 2) return; // 可聴帯域のみ
      const osc = this.ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = f;
      const g = this.ctx.createGain();
      /* ガウス包絡で中央帯域が最も大きく: exp(-(log2(f/center))^2/width^2) */
      const amp = Math.exp(-Math.pow(Math.log2(f / this.center) / this.width, 2));
      g.gain.value = amp;
      osc.connect(g).connect(this.gain);
      osc.start();
      this.oscs.push(osc);
    });
    /* 音量を滑らかに上げる */
    this.gain.gain.setTargetAtTime(0.1, this.ctx.currentTime, 0.02);
  }
}
const shepard = new ShepardTone();

/*-------------------------------------------------------------*
 * 2. 矩形波プレイヤ – 1 本だけ持って切替
 *-------------------------------------------------------------*/
const oscCtx = new AudioContext();            // 別 AudioContext
const osc = oscCtx.createOscillator();
osc.type = "square";
const gain = oscCtx.createGain();
gain.gain.value = 0;                          // 初期は無音
osc.connect(gain).connect(oscCtx.destination);
osc.start();

let prevPC = -1;
function playSquare(pc: number): void {
  if (!useSquare()) {
    /* ←トグルが OFF のときは必ず音量 0 に */
    gain.gain.setTargetAtTime(0, oscCtx.currentTime, 0.02);
    prevPC = -1;                    // 状態もリセット
    return;
  }
  if (pc === prevPC) return;                  // 同じ音なら何もしない
  prevPC = pc;

  const pcToFreq = (p: number) =>
    440 * Math.pow(2, (p - 9) / 12);          // 0=C4≈261.6Hz, 9=A4=440Hz基準
  gain.gain.setTargetAtTime(0.1, oscCtx.currentTime, 0.01); // 音量 0.2
  osc.frequency.setValueAtTime(pcToFreq(pc), oscCtx.currentTime);
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

  const top = topPitchClass(chroma);
  playSquare(top);
  shepard.play(top);
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

function createMeydaChroma(audioCtx: AudioContext, src: MediaStreamAudioSourceNode, cb: (d: Float32Array) => void): MeydaAnalyzer {
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
        const mags = usePercFilter() ? medianFilter(freqBuffer) : freqBuffer;
        drawChroma(accumulateChroma(mags, audioContext!.sampleRate));
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
  history.length = 0;
  if (animationId !== null) cancelAnimationFrame(animationId);
  if (meydaAnalyzer) { meydaAnalyzer.stop(); meydaAnalyzer = null; }
  if (analyser) { analyser.disconnect(); analyser = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

startBtn.addEventListener("click", () => void startCapture());
stopBtn.addEventListener("click", stopCapture);
