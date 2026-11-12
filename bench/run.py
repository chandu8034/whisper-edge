import sys, time, csv, os, torch, soundfile as sf, jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from data import load_librispeech
from energy import EnergyMeter

variant = sys.argv[1] if len(sys.argv) > 1 else "fp32"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MID = "openai/whisper-small"
dtype = torch.float16 if "fp16" in variant else torch.float32

proc = WhisperProcessor.from_pretrained(MID)
model = WhisperForConditionalGeneration.from_pretrained(MID, torch_dtype=dtype).cuda().eval()
items = load_librispeech("../data/LibriSpeech/test-clean", limit)
audios = [(sf.read(p, dtype="float32")[0], ref) for p, ref in items] # file I/O not timed

def transcribe(audio):
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype)
    with torch.inference_mode():
        ids = model.generate(feats, language="en", task="transcribe")
    return proc.batch_decode(ids, skip_special_tokens=True)[0]

for a, _ in audios[:3]:
    transcribe(a)
torch.cuda.reset_peak_memory_stats() # warm-up

norm = proc.tokenizer.normalize
refs, hyps, lat, audio_s = [], [], [], 0.0
meter = EnergyMeter(); meter.start()
for a, ref in audios:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    hyp = transcribe(a)
    torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
    audio_s += len(a) / 16000
    refs.append(norm(ref)); hyps.append(norm(hyp))
gpu_j, cpu_j = meter.stop()

lat.sort(); total = sum(lat); amin = audio_s / 60
row = dict(variant=variant, n=len(lat),
           wer=round(100 * jiwer.wer(refs, hyps), 2),
           p50_ms=round(1000 * lat[len(lat) // 2], 1),
           p95_ms=round(1000 * lat[int(len(lat) * 0.95)], 1),
           rtfx=round(audio_s / total, 1),
           gpu_j_per_min=round(gpu_j / amin, 2),
           cpu_j_per_min=round(cpu_j / amin, 2) if cpu_j is not None else "",
           peak_vram_mb=torch.cuda.max_memory_allocated() // 2**20)
print(row)
os.makedirs("../results", exist_ok=True)
path = "../results/results.csv"; new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if new: w.writeheader()
    w.writerow(row)
