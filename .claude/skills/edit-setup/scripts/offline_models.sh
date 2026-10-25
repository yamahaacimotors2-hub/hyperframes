#!/usr/bin/env bash
# Fetch every model the loadout needs from GitHub releases + npm only — for networks
# where HuggingFace, dl.fbaipublicfiles.com and openaipublic are blocked.
#
#   Whisper small.en (ONNX, sherpa-onnx)        → transcript QA
#   Parakeet TDT 0.6B v2 (ONNX, sherpa-onnx)    → word timestamps
#   Kokoro v1.0 multi-lang (ONNX)               → voiceover (bm_lewis = id 27, am_onyx = 17)
#   DCLAP audio tower + LAION CLAP text tower   → CLAP zero-shot sound ID
#     roberta-base tokenizer extracted from spaCy's en_core_web_trf wheel
#   Demucs htdemucs (ONNX, npm "demucs")        → stem separation
set -euo pipefail
HOME_DIR="${EDIT_SETUP_HOME:-$HOME/.edit-setup}"
M="$HOME_DIR/models"
PY="${EDIT_SETUP_PY:-$HOME_DIR/venv/bin/python}"
mkdir -p "$M"
say() { printf '\033[1;33m◆\033[0m %s\n' "$*"; }
gh_tar() { # url dest-dir-name
  [ -d "$M/$2" ] && { echo "  cached $2"; return; }
  curl -fsSL --retry 3 "$1" | tar xj -C "$M"
}

"$PY" -m pip install -q sherpa-onnx onnxruntime msgpack "transformers>=4.40" librosa soundfile

say "Whisper (ONNX)"
gh_tar https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2 sherpa-onnx-whisper-small.en
say "Parakeet TDT (word timestamps)"
gh_tar https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2 sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
say "Kokoro v1.0 TTS"
gh_tar https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2 kokoro-multi-lang-v1_0

say "CLAP (DCLAP)"
mkdir -p "$M/dclap"
for f in clap_text_model.onnx model_epoch_36.onnx model_epoch_36.onnx.data; do
  [ -s "$M/dclap/$f" ] || curl -fsSL --retry 3 -o "$M/dclap/$f" "https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1/$f"
done
if [ ! -s "$M/dclap/tokenizer/vocab.json" ]; then
  tmp="$(mktemp -d)"
  curl -fsSL --retry 3 -o "$tmp/trf.whl" https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.4.1/en_core_web_trf-3.4.1-py3-none-any.whl
  "$PY" -I - "$tmp/trf.whl" "$M/dclap/tokenizer" <<'EOF'
import os, sys, zipfile, msgpack
whl, out = sys.argv[1], sys.argv[2]
blob = zipfile.ZipFile(whl).read("en_core_web_trf/en_core_web_trf-3.4.1/transformer/model")
shim = msgpack.unpackb(msgpack.unpackb(blob, raw=False, strict_map_key=False)["shims"][1][0], raw=False, strict_map_key=False)
os.makedirs(out, exist_ok=True)
for name, data in shim["tokenizer"].items():
    open(os.path.join(out, name), "wb").write(data)
print("  roberta-base tokenizer →", out)
EOF
  rm -rf "$tmp"
fi

say "Demucs (ONNX via npm)"
mkdir -p "$HOME_DIR/demucs-js"
[ -x "$HOME_DIR/demucs-js/node_modules/.bin/demucs" ] || (cd "$HOME_DIR/demucs-js" && npm init -y >/dev/null && npm install demucs --silent)

say "Done → $M"
