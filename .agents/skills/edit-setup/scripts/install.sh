#!/usr/bin/env bash
# edit-setup installer — the 5-tool Claude Code video-editing loadout:
#   01 HyperFrames (project runtime)   02 Whisper (speech → text)
#   03 CLAP (sound ID)                 04 Demucs (stem split)
#   05 Blender (free 3D renders)
#
# Usage: bash install.sh [--prefetch] [--cpu]
#   --prefetch  download model weights now (Demucs htdemucs, CLAP, Whisper)
#   --cpu       force CPU-only PyTorch wheels (smaller download on Linux)
#
# Idempotent: re-running only installs what is missing.
set -euo pipefail

PREFETCH=0
CPU=0
for a in "$@"; do
  case "$a" in
    --prefetch) PREFETCH=1 ;;
    --cpu) CPU=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

HOME_DIR="${EDIT_SETUP_HOME:-$HOME/.edit-setup}"
VENV="$HOME_DIR/venv"
OS="$(uname -s)"
say() { printf '\033[1;33m◆\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

mkdir -p "$HOME_DIR"

# ── System deps: ffmpeg, Node ≥ 22, Python ≥ 3.10, Blender ───────────────────
if [ "$OS" = "Darwin" ]; then
  have brew || { echo "Install Homebrew first: https://brew.sh" >&2; exit 1; }
  have ffmpeg || brew install ffmpeg
  have node || brew install node
  have python3 || brew install python@3.12
  have cmake || brew install cmake # whisper.cpp builds from source on first use
  if ! have blender && [ ! -d /Applications/Blender.app ]; then brew install --cask blender; fi
elif [ "$OS" = "Linux" ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  if have apt-get; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq ffmpeg python3-venv python3-pip cmake build-essential git
    # Distro Blender lags; prefer snap (current LTS) when available.
    if ! have blender; then
      if have snap; then $SUDO snap install blender --classic; else $SUDO apt-get install -y -qq blender; fi
    fi
  else
    echo "Non-apt Linux: install ffmpeg, python3-venv, cmake, blender manually, then re-run." >&2
  fi
  have node || { echo "Install Node ≥ 22 (https://nodejs.org) and re-run." >&2; exit 1; }
fi

# ── 01 HyperFrames (project runtime) + skills ────────────────────────────────
say "01 HyperFrames"
npx --yes hyperframes@latest --version >/dev/null
npx --yes hyperframes@latest skills update # core skills; workflows install on demand
# Remotion is the alternative runtime from the video. HyperFrames is the default
# here; port Remotion projects with /remotion-to-hyperframes.

# ── Python venv for Demucs + CLAP ────────────────────────────────────────────
say "Python venv → $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip wheel
if [ "$CPU" = 1 ] && [ "$OS" = "Linux" ]; then
  "$PY" -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -m pip install -q torch torchaudio
fi

# ── 02 Whisper ───────────────────────────────────────────────────────────────
# HyperFrames ships whisper.cpp integration (`hyperframes transcribe`); it builds
# whisper.cpp + fetches the ggml model on first use. On Apple Silicon, Parakeet
# is faster and more accurate and is picked automatically when present.
say "02 Whisper"
if [ "$OS" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  "$PY" -m pip install -q parakeet-mlx || true
fi

# ── 03 CLAP (zero-shot audio classification) ─────────────────────────────────
say "03 CLAP"
"$PY" -m pip install -q "transformers>=4.40" soundfile librosa numpy

# ── 04 Demucs (stem separation) ──────────────────────────────────────────────
say "04 Demucs"
"$PY" -m pip install -q demucs

# ── 05 Blender ───────────────────────────────────────────────────────────────
say "05 Blender"
BLENDER="$(command -v blender || true)"
[ -z "$BLENDER" ] && [ -x /Applications/Blender.app/Contents/MacOS/Blender ] && BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
[ -n "$BLENDER" ] || echo "Blender not found — install from https://www.blender.org/download/" >&2

# Persist resolved paths for pipeline.sh / doctor.sh
cat > "$HOME_DIR/env.sh" <<EOF
export EDIT_SETUP_PY="$PY"
export EDIT_SETUP_BLENDER="$BLENDER"
EOF

if [ "$PREFETCH" = 1 ]; then
  say "Prefetching model weights"
  "$PY" -c "from demucs.pretrained import get_model; get_model('htdemucs')"
  "$PY" -c "from transformers import ClapModel, ClapProcessor as P; m='laion/clap-htsat-unfused'; ClapModel.from_pretrained(m); P.from_pretrained(m)"
  tmp="$(mktemp -d)"; ffmpeg -v error -f lavfi -i "sine=f=440:d=2" "$tmp/t.wav"
  npx --yes hyperframes@latest transcribe "$tmp/t.wav" --dir "$tmp" --engine whisper >/dev/null 2>&1 || true
  rm -rf "$tmp"
fi

say "Done. Verify with: bash $(dirname "$0")/doctor.sh"
