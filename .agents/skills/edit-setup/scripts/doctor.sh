#!/usr/bin/env bash
# edit-setup doctor — prints a 5/5 loadout check. Exit 1 if anything required is missing.
set -uo pipefail
HOME_DIR="${EDIT_SETUP_HOME:-$HOME/.edit-setup}"
# shellcheck disable=SC1091
[ -f "$HOME_DIR/env.sh" ] && . "$HOME_DIR/env.sh"
PY="${EDIT_SETUP_PY:-$HOME_DIR/venv/bin/python}"
BLENDER="${EDIT_SETUP_BLENDER:-$(command -v blender || true)}"

n=0; fail=0
ok() { printf '  \033[32m✓\033[0m %-26s %s\n' "$1" "$2"; n=$((n + 1)); }
no() { printf '  \033[31m✗\033[0m %-26s %s\n' "$1" "$2"; fail=1; }
check() { local label="$1" hint="$2"; shift 2; if out="$("$@" 2>/dev/null)"; then ok "$label" "${out%%$'\n'*}"; else no "$label" "$hint"; fi; }

echo "VIDEO EDITING · LOADOUT"
check "ffmpeg" "install ffmpeg" sh -c 'ffmpeg -version | head -1 | cut -d" " -f1-3'
check "01 HyperFrames" "npx hyperframes@latest --version" npx --yes hyperframes@latest --version
check "02 Whisper (transcribe)" "runs via hyperframes transcribe" sh -c 'npx --yes hyperframes@latest transcribe --help >/dev/null && echo "hyperframes transcribe (whisper.cpp / parakeet)"'
check "03 CLAP" "run install.sh" "$PY" -c 'import transformers; print("transformers", transformers.__version__)'
check "04 Demucs" "run install.sh" "$PY" -c 'import demucs; print("demucs", demucs.__version__)'
if [ -n "$BLENDER" ]; then
  check "05 Blender" "blender not runnable" sh -c "'$BLENDER' -b --version | head -1"
else
  no "05 Blender" "install Blender (optional unless the edit needs 3D)"
fi
echo "$((n - 1))/5 tools ready"
exit $fail
