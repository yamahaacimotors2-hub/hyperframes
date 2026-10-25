#!/usr/bin/env bash
# Encode Blender PNG sequences → 1080x1920 H.264 shots, padded to their slot length.
# A shot whose frames are not all rendered yet gets a dark placeholder (marked in its name).
set -euo pipefail
cd "$(dirname "$0")/renders"

# shot:frames:slot-seconds
for spec in reveal:90:3 tank:75:2.5 smg:105:3.5 launch:75:2.5 profile:90:3 hero:90:4.5; do
  IFS=: read -r name frames slot <<<"$spec"
  have=$( (ls "$name"/*.png 2>/dev/null || true) | wc -l)
  if [ "$have" -ge "$frames" ]; then
    [ "$name.mp4" -nt "$name/$(printf '%04d' "$frames").png" ] && { echo "cached  $name"; continue; }
    hold=$(python3 -c "print(max(0, $slot - $frames / 30))")
    ffmpeg -v error -y -framerate 30 -i "$name/%04d.png" \
      -vf "scale=1080:1920:flags=lanczos,unsharp=5:5:0.55:5:5:0,tpad=stop_mode=clone:stop_duration=$hold,format=yuv420p" \
      -c:v libx264 -preset slow -crf 14 -r 30 -movflags +faststart "$name.mp4"
    echo "encoded $name ($frames frames + ${hold}s hold)"
  else
    ffmpeg -v error -y -f lavfi -i "color=c=0x0a1220:s=1080x1920:r=30:d=$slot" \
      -vf "drawtext=text='$name ($have/$frames)':fontcolor=white@0.4:fontsize=60:x=(w-tw)/2:y=(h-th)/2,format=yuv420p" \
      -c:v libx264 -crf 30 "$name.mp4"
    echo "placeholder $name ($have/$frames frames rendered)"
  fi
done
