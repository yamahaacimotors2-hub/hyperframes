"""CLAP zero-shot sound ID over an audio stem.

Slides a window across the audio, scores each window against a label vocabulary
with LAION CLAP, and merges consecutive same-label windows into events.

  python classify_sounds.py effects.wav -o sound-events.json \
      [--labels labels.txt] [--win 1.0] [--hop 0.5] [--min-score 0.35]

Output: {"labels": [...], "events": [{"label", "start", "end", "score"}]}
"""

import argparse
import json
import sys

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

MODEL = "laion/clap-htsat-unfused"
SR = 48_000  # CLAP's native rate

DEFAULT_LABELS = [
    "door slam", "music hit", "whoosh", "impact", "applause", "laughter",
    "footsteps", "keyboard typing", "mouse click", "phone notification",
    "glass breaking", "crowd cheering", "drum roll", "riser", "bass drop",
    "car engine", "dog barking", "wind", "rain", "silence",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="sound-events.json")
    ap.add_argument("--labels", help="text file, one label per line")
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.35)
    args = ap.parse_args()

    labels = DEFAULT_LABELS
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    win, hop = int(args.win * SR), int(args.hop * SR)
    starts = list(range(0, max(1, len(y) - win + 1), hop))
    clips = [y[s : s + win] for s in starts]
    # Skip near-silent windows: CLAP will still pick *some* label for them.
    rms = np.array([float(np.sqrt(np.mean(c**2))) if len(c) else 0.0 for c in clips])
    gate = max(1e-4, float(np.percentile(rms, 20)) * 1.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(MODEL).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL)
    prompts = [f"the sound of {label}" for label in labels]

    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        text_emb = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)

        hits = []
        for i in range(0, len(clips), 32):
            batch = clips[i : i + 32]
            try:
                a = proc(audio=batch, sampling_rate=SR, return_tensors="pt").to(device)
            except TypeError:  # transformers < 4.46 named it `audios`
                a = proc(audios=batch, sampling_rate=SR, return_tensors="pt").to(device)
            audio_emb = torch.nn.functional.normalize(model.get_audio_features(**a), dim=-1)
            probs = (audio_emb @ text_emb.T * model.logit_scale_a.exp()).softmax(-1).cpu().numpy()
            for j, p in enumerate(probs):
                k = i + j
                best = int(p.argmax())
                if rms[k] < gate or p[best] < args.min_score or labels[best] == "silence":
                    continue
                hits.append((starts[k] / SR, (starts[k] + win) / SR, labels[best], float(p[best])))

    events = []
    for s, e, label, score in hits:
        if events and events[-1]["label"] == label and s <= events[-1]["end"]:
            events[-1]["end"] = round(e, 3)
            events[-1]["score"] = round(max(events[-1]["score"], score), 3)
        else:
            events.append({"label": label, "start": round(s, 3), "end": round(e, 3), "score": round(score, 3)})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "labels": labels, "events": events}, f, indent=2)
    print(f"{len(events)} sound events → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
