"""Fuse transcript + silences + sound events into brief.md / analysis.json.

  python make_brief.py <workdir>

Reads <workdir>/{probe.json,transcript.json,silences.txt,sound-events.json}.
Stdlib only, so it runs with any python3.
"""

import json
import re
import sys
from pathlib import Path

FILLERS = {"um", "uh", "erm", "ah", "like", "basically", "literally", "actually", "so", "you know"}
HARD_FILLERS = {"um", "uh", "erm", "ah"}  # always safe to cut; the rest need context


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
    wd = Path(sys.argv[1])
    probe = load(wd / "probe.json", {})
    words = load(wd / "transcript.json", [])
    if isinstance(words, dict):
        words = words.get("words", [])
    events = load(wd / "sound-events.json", {}).get("events", [])
    duration = float(probe.get("format", {}).get("duration", 0) or 0)

    silences = []
    if (wd / "silences.txt").exists():
        start = None
        for line in (wd / "silences.txt").read_text().splitlines():
            if m := re.search(r"silence_start: ([\d.]+)", line):
                start = float(m.group(1))
            elif (m := re.search(r"silence_end: ([\d.]+)", line)) and start is not None:
                silences.append({"start": round(start, 3), "end": round(float(m.group(1)), 3)})
                start = None

    def norm(w: str) -> str:
        return re.sub(r"[^a-z']", "", w.lower())

    hard = [w for w in words if norm(w.get("text", "")) in HARD_FILLERS]
    soft = [w for w in words if norm(w.get("text", "")) in FILLERS - HARD_FILLERS]
    speech = sum(max(0.0, w["end"] - w["start"]) for w in words if "end" in w)
    wpm = round(len(words) / (speech / 60), 1) if speech else 0
    dead_air = [s for s in silences if s["end"] - s["start"] >= 0.6]

    # Sentence boundaries = candidate cut points (word ending in . ! ?).
    beats = [round(w["end"], 3) for w in words if re.search(r"[.!?]$", w.get("text", ""))]

    analysis = {
        "duration": duration,
        "words": len(words),
        "wpm": wpm,
        "dead_air": dead_air,
        "hard_fillers": [{"text": w["text"], "start": w["start"], "end": w["end"]} for w in hard],
        "soft_fillers": [{"text": w["text"], "start": w["start"], "end": w["end"]} for w in soft],
        "sentence_ends": beats,
        "sound_events": events,
    }
    (wd / "analysis.json").write_text(json.dumps(analysis, indent=2))

    text = " ".join(w.get("text", "") for w in words)
    cut = sum(s["end"] - s["start"] for s in dead_air) + sum(w["end"] - w["start"] for w in hard)
    md = [
        f"# Edit brief — {wd.name}",
        "",
        f"- Duration: **{duration:.1f}s** · {len(words)} words · **{wpm} wpm** (target 160–190 for short-form)",
        f"- Dead air ≥0.6s: **{len(dead_air)}** gaps · hard fillers: **{len(hard)}** · soft fillers to review: **{len(soft)}**",
        f"- Tightening potential: ~{cut:.1f}s → ~{max(0.0, duration - cut):.1f}s",
        f"- Sound events (CLAP): {len(events)} — " + ", ".join(sorted({e['label'] for e in events})) if events else "- Sound events (CLAP): none",
        "",
        "## Transcript",
        "",
        text or "_(no transcript)_",
        "",
        "## Files",
        "",
        "- `vocals.wav` / `no_vocals.wav` — Demucs stems (mix voice and bed separately)",
        "- `transcript.json` — word timestamps (from the clean vocal stem)",
        "- `sound-events.json` — CLAP-labelled events on the non-vocal stem",
        "- `analysis.json` — machine-readable version of this brief",
    ]
    (wd / "brief.md").write_text("\n".join(md) + "\n")
    print((wd / "brief.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
