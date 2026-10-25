"""Sound design for "Silence is the new power".

  ~/.edit-setup/venv/bin/python sound.py

1. Synthesize 3–4 variants per SFX slot; CLAP (DCLAP audio + LAION text tower)
   ranks them against a text prompt and keeps the best  → sfx/<slot>.wav, clap.json
2. Compose the score (music bus only)                   → score.wav
3. Demucs (htdemucs ONNX) splits the score              → stems/{drums,bass,other}.wav
   Drum-stem onsets become the visual hit grid           → hits.json
4. Mix: VO (EQ + compression) over a stem-aware duck (melodic stems dip under
   the voice, drums stay), SFX on top                   → mix.wav
5. CLAP labels the finished mix as a QA pass             → clap.json["mix_events"]
"""

import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy import signal
from transformers import AutoTokenizer

HERE = Path(__file__).parent
MODELS = Path.home() / ".edit-setup" / "models"
SR = 44100  # Demucs works at 44.1 kHz; the whole bus runs there
DUR = 25.0
N = int(SR * DUR)
rng = np.random.default_rng(7)

# Timeline (seconds) — cuts land on VO words; see vo.json for word times.
T = dict(
    vo1=0.7, vo2=3.3, shutoff=5.0, green=5.85, whir=5.5, ignite=6.0, drl=6.70,
    vo3=7.9, cut_tank=9.0, tft=9.77, vo4=11.24, cut_smg=11.5, riser=13.2,
    launch=15.0, cut_profile=17.5, vo5=17.7, cut_hero=20.5, sting=21.8, end=25.0,
)
BPM = 120
BEAT = 60 / BPM


# ── DSP helpers ──────────────────────────────────────────────────────────────
def t_(dur):
    return np.arange(int(dur * SR)) / SR


def lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos"), x)


def saw(f, t, voices=1, detune_cents=0.0, phase_seed=0):
    r = np.random.default_rng(phase_seed)
    out = np.zeros_like(t)
    for v in range(voices):
        c = 0 if voices == 1 else detune_cents * (2 * v / (voices - 1) - 1)
        out += signal.sawtooth(2 * np.pi * (f * 2 ** (c / 1200)) * t + r.uniform(0, 6.28))
    return out / voices


def sweep_sin(f0, f1, dur, curve="exp"):
    t = t_(dur)
    f = f0 * (f1 / f0) ** (t / dur) if curve == "exp" else f0 + (f1 - f0) * t / dur
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def env(n, a=0.005, d=0.2, s=0.0, r=0.05, hold=0.0):
    a_, d_, h_, r_ = (int(v * SR) for v in (a, d, hold, r))
    e = np.concatenate([np.linspace(0, 1, max(a_, 1)), np.linspace(1, s, max(d_, 1)), np.full(h_, s), np.linspace(s, 0, max(r_, 1))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def expdecay(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def reverb_ir(rt60=2.2, seed=1):
    r = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = r.standard_normal((2, n)) * np.exp(-6.9 * t / rt60)
    ir = np.stack([lp(ch, 7000) for ch in ir])
    ir[:, : int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))  # pre-delay softness
    return ir / np.abs(ir).sum(axis=1, keepdims=True) * 40


IR_HALL, IR_ROOM = reverb_ir(2.8, 1), reverb_ir(0.9, 2)


def verb(x, ir, wet=0.3):
    """mono or stereo in → stereo out (dry + wet)."""
    st = np.stack([x, x]) if x.ndim == 1 else x
    w = np.stack([signal.fftconvolve(st[i], ir[i])[: st.shape[1]] for i in range(2)])
    return st + wet * w


def pan(x, p):
    """p in [-1, 1] (scalar or array) → stereo, equal power."""
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)])


def drive(x, k=2.0):
    return np.tanh(k * x) / np.tanh(k)


def norm(x, peak=0.9):
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N))

    def add(self, sig, at, gain_db=0.0, p=0.0):
        st = pan(sig, p) if sig.ndim == 1 else sig
        i = int(at * SR)
        if i >= N:
            return
        st = st[:, : N - i]
        self.x[:, i : i + st.shape[1]] += st * 10 ** (gain_db / 20)


# ── SFX generators (variant v changes character; CLAP chooses) ───────────────
def sfx_engine_idle(v, dur=4.9, shut=True):
    rate = [11.5, 12.5, 13.5, 12.0][v]
    tone = [700, 1000, 1400, 850][v]
    n = int(dur * SR)
    out = np.zeros(n + SR)
    t, k = 0.0, 0
    while t < dur:
        r = rate
        if shut and t > dur - 0.42:  # Stop & Start: revs fall away and the engine stops
            r = rate * max(0.35, (dur - t) / 0.42)
        i = int(t * SR)
        m = int(0.07 * SR)
        thump = np.sin(2 * np.pi * (62 + 6 * (k % 2)) * t_(0.07)) * expdecay(m, 0.022)
        burst = lp(rng.standard_normal(m), tone) * expdecay(m, 0.01) * 0.6
        amp = 1.0 if not (shut and t > dur - 0.42) else max(0.15, (dur - t) / 0.42)
        out[i : i + m] += (thump + burst) * amp * (0.9 + 0.2 * rng.random())
        t += (1 / r) * (1 + 0.04 * rng.standard_normal())
        k += 1
    out = bp(out, 40, 2200) + 0.4 * lp(out, 160)
    if shut:
        click = hp(rng.standard_normal(int(0.012 * SR)), 2500) * expdecay(int(0.012 * SR), 0.002)
        ci = int((dur + 0.06) * SR)
        out[ci : ci + len(click)] += click * 0.5
    return norm(drive(out, 1.5), 0.8)


def sfx_smg_whir(v):
    f0, f1, dur = [(300, 2400, 0.55), (220, 1800, 0.6), (400, 3200, 0.5), (260, 2000, 0.65)][v]
    s = sweep_sin(f0, f1, dur) + 0.35 * sweep_sin(f0 * 2, f1 * 2, dur) + 0.15 * sweep_sin(f0 * 3.01, f1 * 3.01, dur)
    s += 0.08 * bp(rng.standard_normal(len(s)), 1500, 6000)
    e = np.linspace(0, 1, len(s)) ** 1.6
    return norm(s * e, 0.7)


def sfx_engine_rev(v, dur=1.6):
    peak = [42, 48, 38, 45][v]
    bright = [2200, 3000, 1800, 2600][v]
    n = int(dur * SR)
    out = np.zeros(n + SR)
    t = 0.0
    while t < dur:
        u = t / dur
        rate = 14 + (peak - 14) * min(1, u / 0.3) if u < 0.3 else peak - (peak - 30) * (u - 0.3) / 0.7
        i = int(t * SR)
        m = int(0.05 * SR)
        p = np.sin(2 * np.pi * 90 * t_(0.05)) * expdecay(m, 0.012) + lp(rng.standard_normal(m), bright) * expdecay(m, 0.006)
        out[i : i + m] += p
        t += 1 / rate
    out = drive(bp(out, 50, 5000), 2.5) * env(len(out), 0.01, 0.2, 0.8, 0.5, hold=dur - 0.5)
    return norm(out, 0.85)


def sfx_whoosh(v, dur=0.7):
    lo, hi = [(300, 4000), (200, 6000), (500, 3000), (150, 5000)][v]
    n = int(dur * SR)
    x = rng.standard_normal(n)
    # sweep a band-pass by processing in blocks
    out = np.zeros(n)
    blk = 512
    for b in range(0, n, blk):
        u = b / n
        fc = lo * (hi / lo) ** (np.sin(np.pi * u) ** 1.5)
        out[b : b + blk] = bp(x[b : b + blk], fc * 0.6, fc * 1.4, 1)
    e = np.sin(np.pi * np.linspace(0, 1, n)) ** [2, 3, 1.5, 2.5][v]
    return norm(out * e, 0.7)


def sfx_impact(v):
    n = int(2.5 * SR)
    sub = sweep_sin([70, 60, 80, 55][v], [32, 30, 36, 28][v], 2.5) * expdecay(n, [0.6, 0.8, 0.5, 0.9][v])
    crack = lp(rng.standard_normal(n), [2500, 1500, 4000, 2000][v]) * expdecay(n, 0.05)
    return norm(drive(sub + 0.6 * crack, 1.8), 0.95)


def sfx_ui_blip(v):
    notes = [(1318.5, 1760), (1760, 2637), (987.8, 1318.5), (1568, 2093)][v]
    out = np.concatenate([np.sin(2 * np.pi * f * t_(0.09)) * env(int(0.09 * SR), 0.002, 0.08) for f in notes])
    return norm(out + 0.3 * np.sin(2 * np.pi * 2 * notes[1] * t_(len(out) / SR)) * env(len(out), 0.002, 0.15), 0.5)


def sfx_light_snap(v):
    n = int(0.6 * SR)
    zap = signal.sawtooth(2 * np.pi * [120, 90, 150, 110][v] * t_(0.6)) * expdecay(n, 0.05)
    zap = hp(zap, 600)
    click = hp(rng.standard_normal(n), 3000) * expdecay(n, 0.004)
    shimmer = bp(rng.standard_normal(n), 6000, 12000) * expdecay(n, 0.15) * 0.3
    return norm(zap * [0.6, 0.8, 0.4, 0.7][v] + click + shimmer, 0.8)


def sfx_riser(v, dur=1.8):
    n = int(dur * SR)
    tone = sweep_sin([180, 120, 240, 150][v], [1400, 1100, 1800, 1300][v], dur)
    noise = np.zeros(n)
    x = rng.standard_normal(n)
    for b in range(0, n, 1024):
        fc = 400 * (12000 / 400) ** (b / n)
        noise[b : b + 1024] = bp(x[b : b + 1024], fc * 0.7, fc * 1.3, 1)
    e = np.linspace(0, 1, n) ** 2.2
    return norm((0.5 * tone + noise) * e, 0.7)


def sfx_city(v, dur=5.0):
    n = int(dur * SR)
    rumble = lp(np.cumsum(rng.standard_normal(n)) * 0.02, [300, 450, 250, 380][v])
    tyres = bp(rng.standard_normal(n), 600, 2500) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t_(dur))) * 0.08
    out = rumble + tyres
    for at, f, d in [(1.3, 415, 0.35), (2.9, 349, 0.5), (3.6, 440, 0.25)][: 2 + v % 2]:  # distant horns
        i = int(at * SR)
        h = lp(saw(f, t_(d), 3, 12) + saw(f * 1.26, t_(d), 3, 12), 1800) * env(int(d * SR), 0.01, 0.05, 0.8, 0.08, hold=d - 0.15)
        out[i : i + len(h)] += 0.12 * h
    out[-int(0.12 * SR) :] *= np.linspace(1, 0, int(0.12 * SR))  # the city drops out with the engine
    return norm(out, 0.6)


SLOTS = {
    "engine_idle": (sfx_engine_idle, "a motorcycle engine idling"),
    "smg_whir": (sfx_smg_whir, "an electric motor whirring up"),
    "engine_rev": (sfx_engine_rev, "a motorcycle engine revving loudly"),
    "whoosh": (sfx_whoosh, "a fast whoosh swoosh transition sound"),
    "impact": (sfx_impact, "a deep cinematic boom impact"),
    "ui_blip": (sfx_ui_blip, "a digital interface notification chime"),
    "light_snap": (sfx_light_snap, "an electric spark zap"),
    "riser": (sfx_riser, "a rising tension riser sound effect"),
    "city": (sfx_city, "busy city traffic ambience with car horns"),
}


# ── CLAP (DCLAP student audio tower + LAION text tower, ONNX) ────────────────
class Clap:
    SR = 48000

    def __init__(self):
        d = MODELS / "dclap"
        self.tok = AutoTokenizer.from_pretrained(str(d / "tokenizer"))
        self.audio = ort.InferenceSession(str(d / "model_epoch_36.onnx"))
        self.text = ort.InferenceSession(str(d / "clap_text_model.onnx"))

    def embed_audio(self, y, sr):
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=self.SR)
        y = (np.clip(y, -1, 1) * 32767).astype(np.int16) / 32767.0
        seg = np.tile(y, int(np.ceil(480000 / max(len(y), 1))))[:480000].astype(np.float32)  # tile short clips; zero padding swamps them
        mel = librosa.feature.melspectrogram(y=seg, sr=self.SR, n_fft=2048, hop_length=480, win_length=2048, window="hann", center=True, pad_mode="reflect", power=2.0, n_mels=128, fmin=0, fmax=14000)
        lm = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)[None, None].astype(np.float32)
        e = self.audio.run(None, {"mel_spectrogram": lm})[0][0]
        return e / np.linalg.norm(e)

    def embed_text(self, texts):
        k = self.tok(texts, padding="max_length", truncation=True, max_length=77, return_tensors="np")
        out = []
        for i in range(len(texts)):
            e = self.text.run(None, {"input_ids": k["input_ids"][i : i + 1].astype(np.int64), "attention_mask": k["attention_mask"][i : i + 1].astype(np.int64)})[0][0]
            out.append(e / np.linalg.norm(e))
        return np.stack(out)


# ── score (music bus only) ───────────────────────────────────────────────────
def build_score():
    m = Bus()
    A1, E2, A2, C3, E3, B3, C4 = 55.0, 82.41, 110.0, 130.81, 164.81, 246.94, 261.63

    # Act 1 drone: detuned saws, slow filter swell, hard stop at shutoff
    d = T["shutoff"]
    tt = t_(d)
    drone = sum(saw(f, tt, 3, 9, i) for i, f in enumerate((A1, E2, A2)))
    drone = np.concatenate([lp(drone[i : i + 4096], 180 + 260 * (i / len(drone))) for i in range(0, len(drone), 4096)])
    drone *= np.linspace(0, 1, len(drone)) ** 0.7
    m.add(verb(drone * 0.5, IR_HALL, 0.25), 0.0, -14)
    # ticking clock (8ths) and heartbeat sub
    tick = hp(rng.standard_normal(int(0.02 * SR)), 5000) * expdecay(int(0.02 * SR), 0.004)
    for k, at in enumerate(np.arange(1.0, d, BEAT / 2)):
        m.add(tick, at, -28 + 6 * (at / d), p=0.4 if k % 2 else -0.4)
    lub = np.sin(2 * np.pi * 48 * t_(0.25)) * expdecay(int(0.25 * SR), 0.06)
    for at in np.arange(2.0, d - 0.2, 1.0):
        m.add(lub, at, -10)
        m.add(lub * 0.6, at + 0.22, -10)

    # BRAAM on ignite
    def braam(at, notes, gain, length=3.0):
        tt = t_(length)
        b = sum(saw(f, tt, 5, 14, i) for i, f in enumerate(notes))
        cut = np.interp(tt, [0, 0.12, 0.6, length], [300, 2600, 900, 400])
        out = np.zeros_like(b)
        for i in range(0, len(b), 2048):
            out[i : i + 2048] = lp(b[i : i + 2048], cut[i])
        out = drive(out * env(len(out), 0.015, length - 0.5, 0.0, 0.4), 2.2)
        m.add(verb(out, IR_HALL, 0.35), at, gain)

    braam(T["ignite"], (A1, A2, E2, C3), -6)

    # Drive section: kick on beats, bass 16ths, hats 8ths, claps on 2 & 4 after the tank cut
    kick = sweep_sin(150, 45, 0.3) * expdecay(int(0.3 * SR), 0.09)
    kick[: int(0.004 * SR)] += hp(rng.standard_normal(int(0.004 * SR)), 2000) * 0.5
    hat = hp(rng.standard_normal(int(0.04 * SR)), 7000) * expdecay(int(0.04 * SR), 0.008)
    clap = (bp(rng.standard_normal(int(0.2 * SR)), 900, 4000) * expdecay(int(0.2 * SR), 0.04)) + 0.4 * np.sin(2 * np.pi * 200 * t_(0.2)) * expdecay(int(0.2 * SR), 0.03)
    bass_pat = [A1, A1, A1, A1, A1, A1, 65.41, A1, A1, A1, A1, A1, 49.0, 49.0, 55.0, 55.0]
    note = t_(BEAT / 4)

    def bassnote(f):
        return lp(saw(f, note, 2, 8), 420) * env(len(note), 0.003, 0.11, 0.25, 0.02)

    for k, at in enumerate(np.arange(T["ignite"], T["launch"] - 0.1, BEAT / 4)):
        if at > T["riser"] + 1.5:  # drop the bass for the pre-launch suck
            break
        m.add(bassnote(bass_pat[k % 16]), at, -12)
    for k, at in enumerate(np.arange(T["ignite"], T["launch"] - 0.25, BEAT / 2)):
        m.add(hat, at, -24 if k % 2 == 0 else -20, p=0.25)
        if k % 2 == 0:
            m.add(kick, at, -5)
        if k % 4 == 2 and at >= T["cut_tank"]:
            m.add(verb(clap, IR_ROOM, 0.4), at, -14)
    # tom build into launch
    tom = lambda f: np.sin(2 * np.pi * np.cumsum(np.linspace(f * 1.4, f, int(0.25 * SR))) / SR) * expdecay(int(0.25 * SR), 0.08)  # noqa: E731
    for k, at in enumerate(np.arange(T["launch"] - 2.0, T["launch"] - 0.08, BEAT / 4)):
        m.add(verb(tom(95 if k % 2 else 75), IR_ROOM, 0.3), at, -16 + 12 * (k / 16))

    # Launch: bigger braam an octave up, full drums with 8th kicks and toms
    braam(T["launch"], (A2, E3, A2 * 2, C3 * 2), -5, 2.5)
    for k, at in enumerate(np.arange(T["launch"], T["cut_profile"] - 0.1, BEAT / 2)):
        m.add(kick, at, -4)
        m.add(hat, at + BEAT / 4, -20, p=-0.25)
        if k % 2:
            m.add(verb(clap, IR_ROOM, 0.4), at, -12)
        m.add(verb(tom(80), IR_ROOM, 0.3), at + BEAT / 4, -14)
    for k, at in enumerate(np.arange(T["launch"], T["cut_profile"] - 0.1, BEAT / 4)):
        m.add(bassnote(bass_pat[k % 16] * 2), at, -13)

    # Profile: warm add9 pad + soft pulse
    pad_len = T["cut_hero"] - T["cut_profile"] + 0.4
    tt = t_(pad_len)
    pad = lp(sum(saw(f, tt, 3, 10, i) for i, f in enumerate((A2, E3, B3, C4))), 1200) * env(len(tt), 0.6, 0.1, 0.9, 0.5, hold=pad_len - 1.2)
    m.add(verb(pad * 0.5, IR_HALL, 0.45), T["cut_profile"], -15)
    for at in np.arange(T["cut_profile"], T["cut_hero"] - 0.1, BEAT):
        m.add(kick, at, -15)

    # Hero / end: final braam, long chord, bell sting on the tagline
    braam(T["cut_hero"], (A1, A2, E3, C3 * 2), -7, 4.0)
    tail = T["end"] - T["cut_hero"]
    tt = t_(tail)
    chord = lp(sum(saw(f, tt, 4, 12, i) for i, f in enumerate((A2, E3, C4, E3 * 2))), 1500) * env(len(tt), 0.4, 0.1, 0.8, 2.2, hold=tail - 2.8)
    m.add(verb(chord * 0.45, IR_HALL, 0.6), T["cut_hero"], -14)
    bell = sum(a * np.sin(2 * np.pi * f * t_(3.0)) * expdecay(int(3 * SR), tau) for f, a, tau in ((880, 1, 1.2), (1760.4, 0.4, 0.6), (2637, 0.25, 0.35), (3520.8, 0.12, 0.2)))
    m.add(verb(bell * 0.35, IR_HALL, 0.7), T["sting"], -12)
    return m.x


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    (HERE / "sfx").mkdir(exist_ok=True)
    clap = Clap()
    report = {"slots": {}}

    # 1. CLAP-ranked SFX
    picked = {}
    for slot, (gen, prompt) in SLOTS.items():
        cands = [gen(v) for v in range(4)]
        te = clap.embed_text([prompt])[0]
        scores = [float(clap.embed_audio(c, SR) @ te) for c in cands]
        best = int(np.argmax(scores))
        picked[slot] = cands[best]
        sf.write(HERE / "sfx" / f"{slot}.wav", cands[best], SR)
        report["slots"][slot] = {"prompt": prompt, "scores": [round(s, 3) for s in scores], "picked": best}
        print(f"CLAP {slot:12s} {prompt!r:48s} scores={np.round(scores, 3)} → v{best}")

    # 2. score
    score = build_score()
    score = norm(score, 0.89)
    sf.write(HERE / "score.wav", score.T, SR)

    # 3. Demucs stems
    stems_dir = HERE / "stems"
    if not (stems_dir / "drums.wav").exists():
        cli = Path.home() / ".edit-setup" / "demucs-js" / "node_modules" / ".bin" / "demucs"
        subprocess.run([str(cli), str(HERE / "score.wav"), "--output", str(HERE / "demucs-out")], check=True)
        found = {p.stem: p for p in (HERE / "demucs-out").rglob("*.wav")}
        stems_dir.mkdir(exist_ok=True)
        for name in ("drums", "bass", "other", "vocals"):
            key = next((k for k in found if name in k.lower()), None)
            if key:
                found[key].rename(stems_dir / f"{name}.wav")
    stems = {}
    for name in ("drums", "bass", "other", "vocals"):
        p = stems_dir / f"{name}.wav"
        if p.exists():
            y, sr = sf.read(p, always_2d=True)
            y = y.T if sr == SR else np.stack([librosa.resample(c, orig_sr=sr, target_sr=SR) for c in y.T])
            stems[name] = np.pad(y, ((0, 0), (0, max(0, N - y.shape[1]))))[:, :N]
    print("Demucs stems:", {k: round(float(np.sqrt(np.mean(v**2))), 4) for k, v in stems.items()})

    drums_mono = stems["drums"].mean(0)
    onset_env = librosa.onset.onset_strength(y=drums_mono.astype(np.float32), sr=SR, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=512, units="time", backtrack=False)
    strength = onset_env[librosa.time_to_frames(onsets, sr=SR, hop_length=512)]
    hits = [{"t": round(float(t), 3), "s": round(float(s / strength.max()), 3)} for t, s in zip(onsets, strength)]
    (HERE / "hits.json").write_text(json.dumps({"source": "demucs drums stem onsets", "hits": hits}, indent=1))
    print(f"{len(hits)} drum hits → hits.json")

    # 4. VO bus
    vo = json.loads((HERE / "vo.json").read_text())
    vbus = Bus()
    for line in vo["lines"]:
        y, sr = sf.read(HERE / f"{line['id']}.wav")
        y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=SR)
        y = hp(y, 85)
        y = y + 0.35 * bp(y, 2500, 5000) + 0.25 * lp(y, 180)  # presence + chest
        env_ = signal.lfilter([0.002], [1, -0.998], np.abs(y)) * 8 + 1e-6
        y = y * np.minimum(1.0, (0.12 / env_) ** 0.6)  # ~2.5:1 compression above threshold
        vbus.add(verb(norm(y, 0.9), IR_ROOM, 0.08), T[line["id"]], -2)
    vmono = np.abs(vbus.x).mean(0)
    venv = signal.lfilter([0.0015], [1, -0.9985], vmono)
    venv = np.minimum(1.0, venv / (venv.max() * 0.35))
    duck = 1 - 0.62 * venv  # ≈ -8.4 dB on melodic stems under speech
    music = stems["drums"] * (1 - 0.25 * venv) + (stems["bass"] + stems["other"] + stems.get("vocals", 0)) * duck

    # 5. SFX bus
    s = Bus()
    s.add(picked["city"], 0.0, -9)
    s.add(picked["engine_idle"], 0.15, -7, p=-0.1)
    s.add(sfx_ui_blip(2) * 0.6, T["green"], -16, p=0.2)
    s.add(picked["smg_whir"], T["ignite"] - len(picked["smg_whir"]) / SR, -9)
    s.add(picked["impact"], T["ignite"], -3)
    s.add(picked["engine_rev"], T["ignite"] + 0.05, -8)
    s.add(verb(picked["light_snap"], IR_HALL, 0.3), T["drl"], -10)
    for at in (T["cut_tank"], T["cut_smg"], T["cut_profile"]):
        w = picked["whoosh"]
        s.add(w, at - len(w) / SR * 0.55, -12, p=np.linspace(-0.6, 0.6, len(w)))
    s.add(picked["ui_blip"], T["tft"], -14, p=0.1)
    for at in (12.17, 13.47):  # SMG ring pulses on "Generator" and "push"
        z = sfx_smg_whir(1)[: int(0.5 * SR)] * np.hanning(int(0.5 * SR))
        s.add(z, at - 0.1, -16)
    s.add(picked["riser"], T["launch"] - len(picked["riser"]) / SR, -10)
    s.add(picked["impact"], T["launch"], -3)
    rev = sfx_engine_rev(1, 2.0)
    s.add(rev, T["launch"] + 0.05, -7, p=np.linspace(-0.8, 0.8, len(rev)))  # pass-by, left to right
    for at in (15.2, 15.75, 16.3):
        s.add(picked["whoosh"] * 0.8, at, -15, p=np.linspace(0.7, -0.7, len(picked["whoosh"])))
    s.add(picked["impact"] * 0.7, T["cut_hero"], -7)

    mix = vbus.x + music * 10 ** (-3 / 20) + s.x
    mix = drive(mix * 0.9, 1.2) * 0.92
    sf.write(HERE / "mix.wav", mix.T, SR)
    sf.write(HERE / "music_ducked.wav", music.T, SR)
    sf.write(HERE / "sfx_bus.wav", s.x.T, SR)
    sf.write(HERE / "vo_bus.wav", vbus.x.T, SR)

    # 6. CLAP QA over the finished mix (1 s windows)
    labels = ["a motorcycle engine idling", "silence", "a deep cinematic boom impact", "a man speaking", "electronic dance music with drums", "a fast whoosh transition", "cinematic orchestral music", "city traffic ambience"]
    te = clap.embed_text(labels)
    mono = mix.mean(0)
    ev = []
    for i in range(int(DUR)):
        seg = mono[max(0, i * SR - SR // 2) : (i + 1) * SR + SR // 2]  # 2 s window centred on second i
        if np.sqrt(np.mean(seg**2)) < 1e-3:
            ev.append({"t": i, "label": "silence", "score": 1.0})
            continue
        sc = te @ clap.embed_audio(seg, SR)
        ev.append({"t": i, "label": labels[int(sc.argmax())], "score": round(float(sc.max()), 3)})
    report["mix_events"] = ev
    (HERE / "clap.json").write_text(json.dumps(report, indent=1))
    print("CLAP mix QA:", " | ".join(f"{e['t']}s {e['label']}" for e in ev))


if __name__ == "__main__":
    main()
