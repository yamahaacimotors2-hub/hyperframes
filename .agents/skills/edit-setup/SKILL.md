---
name: edit-setup
description: Claude-powered video editing with the 5-tool loadout — HyperFrames (project runtime), Whisper (speech → text), CLAP (sound ID), Demucs (stem split) and Blender (free 3D renders). Use when the user hands over raw footage (talking-head, podcast, vlog, product demo) and wants Claude to edit it end to end — transcribe, tighten, caption, sound-design, add 3D objects and render — or asks to install / verify this editing stack.
---

# edit-setup — the Claude-powered video editing system

Five tools, one loop. Claude never "watches" the video: it **reads** it.
Demucs and Whisper turn the audio into words with timestamps, CLAP turns the
rest of the audio into labelled events, and HyperFrames turns Claude's edit
decisions into HTML it can render deterministically. Blender supplies 3D
objects for free instead of spending generation credits per shot.

| #   | Tool                               | Job in the pipeline                            | Output                            |
| --- | ---------------------------------- | ---------------------------------------------- | --------------------------------- |
| 01  | HyperFrames (or Remotion)          | project runtime: composition, captions, render | `index.html` → MP4                |
| 02  | Whisper (`hyperframes transcribe`) | hear every word                                | `transcript.json`, `captions.srt` |
| 03  | CLAP                               | tell Claude what each sound is                 | `sound-events.json`               |
| 04  | Demucs                             | separate voice / music / effects               | `vocals.wav`, `no_vocals.wav`     |
| 05  | Blender                            | any 3D object, rendered locally, free          | `assets/<name>.webm` (alpha)      |

## 0. Setup gate (once per machine)

```bash
bash .claude/skills/edit-setup/scripts/install.sh --prefetch   # add --cpu on Linux without a GPU
bash .claude/skills/edit-setup/scripts/doctor.sh               # must print 5/5 (Blender may be skipped if no 3D)
```

Don't start an edit until `doctor.sh` passes for every tool the edit needs.

## 1. Brief (ask once, then go)

Confirm in **one** message, defaulting anything the user doesn't care about:

- **Deliverable**: platform + aspect (9:16 Reels/TikTok/Shorts default, 16:9 YouTube, 1:1), target length.
- **Edit depth**: `light` (captions + loudness) · `tight` (also cut dead air & hard fillers) · `produced` (also overlays, SFX, music bed, 3D).
- **Brand**: colors, fonts, handle. If absent, pick one palette and hold it for the whole video.
- **Hook**: the first 2 seconds. If the footage has no hook, propose one from the transcript.

## 2. Analyze (machine pass — no creative decisions yet)

```bash
bash .claude/skills/edit-setup/scripts/pipeline.sh analyze <video> edit-work/<name>
```

Stages, in this order on purpose:

1. **Ingest** → constant-frame-rate `source.mp4` (phone footage is VFR; VFR drifts against word timestamps).
2. **Demucs before Whisper** → transcribe the isolated vocal stem. Music and room noise are what make ASR mishear; removing them first is the single biggest transcript-accuracy win. The stems also let you duck the bed under speech instead of ducking everything.
3. **Whisper** → word-level `transcript.json` + `captions.srt`; `silencedetect` on the vocal stem → dead-air map.
4. **CLAP on the non-vocal stem** → labelled events (door slam, music hit, whoosh…). Override the vocabulary with `CLAP_LABELS=labels.txt` when you know what's in the footage — zero-shot CLAP is only as good as its label list.
5. **Brief** → `brief.md` + `analysis.json`: wpm, dead air, fillers, sentence ends, events.

Read `brief.md` and `transcript.json` before deciding anything. Never cut from memory of the transcript; cut from timestamps.

## 3. Decide (Claude's pass) → `edl.json`

Write an edit decision list next to the brief. It is the contract between analysis and composition, so the user can review it before anything renders:

```json
{
  "keep": [
    { "start": 0.42, "end": 7.88 },
    { "start": 8.61, "end": 15.2 }
  ],
  "hook": { "start": 0.42, "end": 2.3, "treatment": "kinetic title over footage" },
  "captions": { "mode": "rail", "emphasis": ["nobody", "everything", "free"] },
  "overlays": [{ "at": 9.1, "dur": 3.0, "kind": "tool-card", "text": "WHISPER" }],
  "sfx": [{ "at": 9.1, "file": "whoosh", "reason": "overlay enters" }],
  "music": { "bed": "no_vocals.wav | library", "duck_db": -14 },
  "three_d": [{ "name": "cube", "args": "--primitive cube --color #D9653B", "at": 30.5 }]
}
```

Top-1% rules for the decide pass:

- **Cut on sentence ends** (`analysis.sentence_ends`), never mid-word; leave 80–120 ms of tail so speech doesn't clip.
- **Hard fillers** (um/uh) always go; **soft fillers** (like/so/actually) only when the sentence still reads without them.
- **Dead air ≥ 0.6 s** goes unless it's a deliberate beat before a punchline.
- **Something changes every 2–4 s** in short-form: a cut, a caption emphasis, an overlay, a punch-in. Plan it on the timeline; don't sprinkle it.
- **Every overlay is caused by a word.** Anchor each to the timestamp of the word that motivates it (`"first, grab Hyperframe"` → the HyperFrames card lands on "Hyperframe").
- **SFX follow CLAP + motion**: reuse the footage's own hits (CLAP events) as sync points, and add a whoosh/click only where something moves on screen.
- **One accent color, one display face.** Emphasis words take the accent; everything else stays neutral.
- **Close with a single CTA** that's on screen and spoken at the same time.

Show the user the EDL summary (kept length, number of cuts, overlay list) and get a go before building, unless they said not to ask.

## 4. 3D objects (optional)

```bash
bash .claude/skills/edit-setup/scripts/pipeline.sh 3d cube --primitive cube --color "#D9653B"
bash .claude/skills/edit-setup/scripts/pipeline.sh 3d product --model product.glb --frames 120
```

Renders a looping transparent turntable to `assets/<name>.webm`. Layer it as a `<video>` clip over the footage. Free and deterministic — use this before reaching for an AI-generation model.

## 5. Build (HyperFrames)

Load the matching HyperFrames workflow skill and feed it the EDL rather than re-deriving decisions:

- captions only → `/embedded-captions`
- overlays, tool cards, lower-thirds, PiP → `/talking-head-recut`
- composition contract & motion → `/hyperframes-core`, `/hyperframes-animation`, `motion-doctrine`, `captions-overlay`
- music/voice mix (duck the bed under `vocals.wav`) → `/hyperframes-audio`; extra SFX/music → `/media-use`

Use `source.mp4` (not the original) as the footage clip, and `transcript.json` from the work dir as the caption source.

## 6. Finish gate

```bash
bash .claude/skills/edit-setup/scripts/pipeline.sh finish <project-dir> renders/final.mp4
```

Runs `hyperframes lint` → `hyperframes check` → `render` → loudness normalize to −14 LUFS / −1 dBTP. Both lint and check must pass. Then `hyperframes snapshot` the hook, a mid-point and the CTA, look at them, and only then hand over the file.

## Failure modes

- **Captions drift late** → source was VFR; use `source.mp4` from the work dir.
- **Transcript mishears over music** → you transcribed the mix, not `vocals.wav`.
- **CLAP labels everything "music"** → narrow the label list to what's plausible in this footage, raise `--min-score`.
- **Blender render black** → object outside camera; the script normalizes to a 2-unit box, so check imported model scale/units.
- **Model downloads fail** → HuggingFace / `dl.fbaipublicfiles.com` blocked on this network; run `install.sh --prefetch` where they're reachable.
