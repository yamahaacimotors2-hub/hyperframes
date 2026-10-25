"""Voiceover: Kokoro TTS → Whisper QA → Parakeet word timings.

  ~/.edit-setup/venv/bin/python voice.py

Writes vo_<id>.wav (24 kHz mono, trimmed) and vo.json:
  {"lines": [{"id", "text", "dur", "whisper", "match", "words": [{"text","start","end"}]}]}
A line whose Whisper transcript does not match the script fails the run.
"""

import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import sherpa_onnx
import soundfile as sf

MODELS = Path.home() / ".edit-setup" / "models"
HERE = Path(__file__).parent
VOICE, SPEED = 27, 0.86  # bm_lewis — deepest, most expressive male voice measured

LINES = {
    "vo1": "In a city that never goes quiet,",
    "vo2": "it knows when to rest.",
    "vo3": "Stop and Start. It sleeps at every red light.",
    "vo4": "Smart Motor Generator. Silent push, the moment you go.",
    "vo5": "Hybrid. Built for the city you ride.",
}


def norm(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).split()


def main() -> int:
    k = MODELS / "kokoro-multi-lang-v1_0"
    tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(k / "model.onnx"),
                    voices=str(k / "voices.bin"),
                    tokens=str(k / "tokens.txt"),
                    data_dir=str(k / "espeak-ng-data"),
                    dict_dir=str(k / "dict"),
                    lexicon=f"{k}/lexicon-us-en.txt,{k}/lexicon-gb-en.txt",
                ),
                num_threads=4,
            )
        )
    )
    w = MODELS / "sherpa-onnx-whisper-small.en"
    whisper = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(w / "small.en-encoder.int8.onnx"),
        decoder=str(w / "small.en-decoder.int8.onnx"),
        tokens=str(w / "small.en-tokens.txt"),
        num_threads=4,
        language="en",
        task="transcribe",
    )
    p = MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
    parakeet = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(p / "encoder.int8.onnx"),
        decoder=str(p / "decoder.int8.onnx"),
        joiner=str(p / "joiner.int8.onnx"),
        tokens=str(p / "tokens.txt"),
        model_type="nemo_transducer",
        num_threads=4,
    )

    def run(rec, y16):
        s = rec.create_stream()
        s.accept_waveform(16000, y16)
        rec.decode_stream(s)
        return s.result

    out, ok = [], True
    for lid, text in LINES.items():
        a = tts.generate(text, sid=VOICE, speed=SPEED)
        y = np.array(a.samples, dtype=np.float32)
        idx = np.where(np.abs(y) > 0.01)[0]
        y = y[max(0, idx[0] - 240) : idx[-1] + 2400]
        sf.write(HERE / f"{lid}.wav", y, a.sample_rate)
        y16 = librosa.resample(y, orig_sr=a.sample_rate, target_sr=16000)

        heard = run(whisper, y16).text.strip()
        match = norm(heard) == norm(text)
        ok &= match

        # Parakeet tokens are sub-word pieces; a piece starting with a space opens a new word.
        r = run(parakeet, y16)
        words = []
        for tok, t0 in zip(r.tokens, r.timestamps):
            if tok.startswith(" ") or not words:
                words.append({"text": tok.strip(), "start": round(t0, 3)})
            else:
                words[-1]["text"] += tok
        for i, wd in enumerate(words):
            wd["end"] = words[i + 1]["start"] if i + 1 < len(words) else round(len(y) / a.sample_rate, 3)

        out.append({"id": lid, "text": text, "dur": round(len(y) / a.sample_rate, 3), "whisper": heard, "match": match, "words": words})
        print(f"{lid} {'OK ' if match else 'MISMATCH'} {len(y) / a.sample_rate:.2f}s  whisper: {heard!r}")
        print("     " + " ".join(f"{wd['text']}@{wd['start']:.2f}" for wd in words))

    (HERE / "vo.json").write_text(json.dumps({"voice": "kokoro v1.0 bm_lewis", "lines": out}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
