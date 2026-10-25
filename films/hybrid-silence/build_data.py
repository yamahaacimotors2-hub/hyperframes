"""Emit data.js for the composition: word-timed captions (script words + Parakeet
timings, Whisper-verified) and the Demucs drum-hit grid.

  python3 build_data.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
vo = json.loads((HERE / "audio" / "vo.json").read_text())
hits = json.loads((HERE / "audio" / "hits.json").read_text())["hits"]

# Global VO start times — must match audio/sound.py T[...]
START = {"vo1": 0.7, "vo2": 3.3, "vo3": 7.9, "vo4": 11.24, "vo5": 17.7}
# Phrase breaks (index of the first word of each new phrase) and emphasis words
BREAKS = {"vo3": [3], "vo4": [3, 5], "vo5": [1]}
EMPH = {"quiet", "rest", "stop", "start", "red", "light", "smart", "motor", "generator", "silent", "push", "hybrid", "ride"}
ZONE = {"vo1": "mid", "vo2": "mid", "vo3": "low", "vo4": "low", "vo5": "low"}

phrases = []
for line in vo["lines"]:
    assert line["match"], f"{line['id']} failed Whisper QA"
    words = line["text"].split()
    timed = line["words"]
    assert len(words) == len(timed), f"{line['id']}: {len(words)} script words vs {len(timed)} timed"
    cuts = [0, *BREAKS.get(line["id"], []), len(words)]
    for a, b in zip(cuts, cuts[1:]):
        ws = []
        for w, t in zip(words[a:b], timed[a:b]):
            ws.append({
                "w": w,
                "t": round(START[line["id"]] + t["start"], 3),
                "e": re.sub(r"[^a-z]", "", w.lower()) in EMPH,
            })
        end = START[line["id"]] + (timed[b]["start"] if b < len(timed) else line["dur"])
        phrases.append({"id": f"{line['id']}-{a}", "zone": ZONE[line["id"]], "words": ws, "end": round(end, 3)})

# "go." is the launch slam, not a caption word
for p in phrases:
    go = [w for w in p["words"] if w["w"].lower() == "go."]
    if go:
        p["end"] = go[0]["t"]
        p["words"] = [w for w in p["words"] if w not in go]

data = {"phrases": phrases, "hits": [h for h in hits if h["s"] >= 0.35]}
(HERE / "data.js").write_text("window.FILM = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"{len(phrases)} phrases, {len(data['hits'])} hits → data.js")
for p in phrases:
    print(f"  {p['words'][0]['t']:6.2f}–{p['end']:6.2f}  {' '.join(w['w'] for w in p['words'])}")
