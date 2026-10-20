"""Run on your laptop (needs internet): pip install edge-tts
  python make_voice.py samples            -> 30 s samples in several accents to choose from
  python make_voice.py episode <script.md> <out_folder> [voice]  -> narration .mp3 + .srt subtitles
Lines starting with '#' or 'Screen:' or empty are not spoken. Each spoken line becomes one subtitle."""
import asyncio, sys, pathlib, edge_tts

VOICES = {"US": "en-US-AriaNeural", "UK": "en-GB-SoniaNeural", "India": "en-IN-NeerjaNeural", "US-male": "en-US-GuyNeural", "UK-male": "en-GB-RyanNeural"}
SAMPLE = ("Every day, a power plant in Portugal must decide what to sell, before it knows tomorrow's prices. "
          "I built the system that decides. Let me show you, one minute at a time.")

def spoken(path):
    return [l.strip() for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("Screen:")]

def ts(t):
    t = max(t, 0); h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

async def episode(script, out, voice):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    com = edge_tts.Communicate(" ".join(spoken(script)), voice, boundary="SentenceBoundary")
    audio, cues = bytearray(), []
    async for ch in com.stream():
        if ch["type"] == "audio": audio += ch["data"]
        elif ch["type"] == "SentenceBoundary": cues.append((ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]))
    stem = pathlib.Path(script).stem
    (out / f"{stem}.mp3").write_bytes(bytes(audio))
    (out / f"{stem}.srt").write_text("".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1)), encoding="utf-8")
    print("wrote", out / f"{stem}.mp3", "and .srt", f"({cues[-1][1]:.0f} s)")

async def samples():
    for name, v in VOICES.items():
        await edge_tts.Communicate(SAMPLE, v).save(f"sample_{name}.mp3"); print("sample_%s.mp3" % name)

if sys.argv[1] == "samples": asyncio.run(samples())
else: asyncio.run(episode(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else VOICES["US"]))
