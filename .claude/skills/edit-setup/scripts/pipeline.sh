#!/usr/bin/env bash
# edit-setup pipeline. Every stage is cached: an existing output is reused, so
# re-runs are cheap and a failed stage resumes where it stopped.
#
#   pipeline.sh analyze <video> [workdir]      ingest → stems → transcript → sound ID → brief
#   pipeline.sh 3d <out-name> [blender args]   Blender turntable → assets/<out-name>.webm (alpha)
#   pipeline.sh finish <project-dir> [out.mp4] lint → check → render → loudness-normalize
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${EDIT_SETUP_HOME:-$HOME/.edit-setup}"
# shellcheck disable=SC1091
[ -f "$HOME_DIR/env.sh" ] && . "$HOME_DIR/env.sh"
PY="${EDIT_SETUP_PY:-$HOME_DIR/venv/bin/python}"
BLENDER="${EDIT_SETUP_BLENDER:-$(command -v blender || true)}"
HF="npx --yes hyperframes@latest"
step() { printf '\n\033[1;33m[%s]\033[0m %s\n' "$1" "$2"; }
cached() { [ -s "$1" ] && printf '  cached: %s\n' "$1"; }

analyze() {
  local src="$1" wd="${2:-edit-work/$(basename "${1%.*}")}"
  mkdir -p "$wd"

  step 0/4 "Ingest"
  cached "$wd/probe.json" || ffprobe -v error -print_format json -show_format -show_streams "$src" > "$wd/probe.json"
  # Constant frame rate source: phone footage is VFR and drifts against captions.
  cached "$wd/source.mp4" || ffmpeg -v error -y -i "$src" -c:v libx264 -crf 16 -preset fast -r 30 -fps_mode cfr -c:a aac -b:a 256k -movflags +faststart "$wd/source.mp4"
  cached "$wd/audio.wav" || ffmpeg -v error -y -i "$src" -vn -ac 2 -ar 44100 "$wd/audio.wav"

  step 1/4 "04 Demucs — split voice from music/effects (cleaner ASR + separate mixing)"
  if ! cached "$wd/vocals.wav"; then
    "$PY" -m demucs --two-stems=vocals -n htdemucs -o "$wd/demucs" "$wd/audio.wav"
    cp "$wd/demucs/htdemucs/audio/vocals.wav" "$wd/vocals.wav"
    cp "$wd/demucs/htdemucs/audio/no_vocals.wav" "$wd/no_vocals.wav"
  fi

  step 2/4 "02 Whisper — word-level transcript from the vocal stem"
  if ! cached "$wd/transcript.json"; then
    $HF transcribe "$wd/vocals.wav" --dir "$wd" ${HF_LANG:+--language "$HF_LANG"} ${HF_MODEL:+--model "$HF_MODEL"}
  fi
  cached "$wd/captions.srt" || $HF transcribe "$wd/transcript.json" --dir "$wd" --to srt --output "$wd/captions.srt"
  cached "$wd/silences.txt" || ffmpeg -v info -i "$wd/vocals.wav" -af silencedetect=noise=-35dB:d=0.35 -f null - 2> "$wd/silences.txt" || true

  step 3/4 "03 CLAP — identify sounds on the non-vocal stem"
  cached "$wd/sound-events.json" || "$PY" "$HERE/classify_sounds.py" "$wd/no_vocals.wav" -o "$wd/sound-events.json" ${CLAP_LABELS:+--labels "$CLAP_LABELS"}

  step 4/4 "Brief"
  python3 "$HERE/make_brief.py" "$wd"
  echo "→ $wd/brief.md"
}

three_d() {
  local name="$1"; shift
  [ -n "$BLENDER" ] || { echo "Blender not installed (run install.sh)" >&2; exit 1; }
  step 05 "Blender — $name"
  local frames="assets/.render-$name"
  "$BLENDER" -b -P "$HERE/blender_object.py" -- --out "$frames" "$@"
  mkdir -p assets
  ffmpeg -v error -y -framerate 30 -i "$frames/%04d.png" -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 30 -auto-alt-ref 0 "assets/$name.webm"
  echo "→ assets/$name.webm (transparent; layer it as a <video> clip)"
}

finish() {
  local dir="$1" out="${2:-$1/renders/final.mp4}"
  step 1/4 "hyperframes lint";  (cd "$dir" && $HF lint)
  step 2/4 "hyperframes check"; (cd "$dir" && $HF check)
  step 3/4 "hyperframes render"
  mkdir -p "$(dirname "$out")"
  out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
  local raw="${out%.mp4}.raw.mp4"
  (cd "$dir" && $HF render --output "$raw")
  step 4/4 "Loudness → -14 LUFS / -1 dBTP (platform target)"
  ffmpeg -v error -y -i "$raw" -c:v copy -af loudnorm=I=-14:TP=-1:LRA=11 -c:a aac -b:a 256k "$out"
  rm -f "$raw"
  echo "→ $out"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  analyze) analyze "$@" ;;
  3d) three_d "$@" ;;
  finish) finish "$@" ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
