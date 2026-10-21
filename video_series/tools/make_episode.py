"""One command per episode (run on your laptop, dashboard running at http://localhost:8501):
  pip install edge-tts playwright imageio-ffmpeg   (once)   and   playwright install chromium   (once)
  python make_episode.py <script.md> <output_folder> [--voice en-US-AriaNeural] [--size 1080] [--keep-sidebar] [--fake-voice]

Script format (see scripts/*.md): blocks separated by an empty line. A block may start with
  Screen: <note for you, not used>
  Do: scroll 0.3 | click Gate button text | wait 2      (several actions separated by '|', run at the start of the block)
and then the spoken sentences (one per line; each becomes one subtitle). Lines starting with '#' are ignored.
Output: episode.mp4 (square, burned-in subtitles, AI voice), episode.srt, episode.mp3."""
import argparse, asyncio, pathlib, re, subprocess, sys, time

try:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

GAP = 0.45          # seconds of silence between sentences
LEAD = 1.2          # seconds of picture before the voice starts
TAIL = 1.5          # seconds of picture after the last sentence

def parse(path):
    blocks, cur = [], None
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        l = raw.strip()
        if not l:
            cur = None; continue
        if l.startswith("#"): continue
        if cur is None:
            cur = {"do": [], "say": []}; blocks.append(cur)
        if l.lower().startswith("screen:"): continue
        if l.lower().startswith("do:"):
            cur["do"] = [a.strip() for a in l[3:].split("|") if a.strip()]
        else:
            cur["say"].append(l)
    return [b for b in blocks if b["say"] or b["do"]]

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def duration(f):
    r = run([FFMPEG, "-i", str(f), "-f", "null", "-"])
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", r.stderr)
    h, mi, s = m[-1]; return int(h) * 3600 + int(mi) * 60 + float(s)

async def make_audio(sentences, out, voice, fake):
    """One mp3 per sentence (exact durations), then joined with gaps into narration.mp3. Returns [(start, end)] in seconds."""
    parts, spans, t = [], [], 0.0
    sil = out / "gap.mp3"
    run([FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(GAP), "-c:a", "libmp3lame", "-b:a", "48k", str(sil)])
    for i, s in enumerate(sentences):
        f = out / f"s{i:03}.mp3"
        if fake:   # silent placeholder, about 0.38 s per word, only for testing the picture
            run([FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{0.38 * len(s.split()):.2f}", "-c:a", "libmp3lame", "-b:a", "48k", str(f)])
        else:
            import edge_tts
            await edge_tts.Communicate(s, voice).save(str(f))
        d = duration(f); spans.append((t, t + d)); t += d + GAP; parts += [f, sil]
    lst = out / "list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "libmp3lame", "-b:a", "96k", str(out / "narration.mp3")])
    return spans

def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

SUB_JS = """(t) => { let d = document.getElementById('__sub');
 if (!d) { d = document.createElement('div'); d.id = '__sub';
  d.style.cssText = 'position:fixed;left:4%;right:4%;bottom:5%;z-index:99999;background:rgba(0,0,0,.78);color:#fff;font:600 34px/1.3 Arial,sans-serif;padding:14px 22px;border-radius:14px;text-align:center;';
  document.body.appendChild(d); }
 d.textContent = t; d.style.display = t ? 'block' : 'none'; }"""
SCROLL_JS = """(f) => { const els = [...document.querySelectorAll('[data-testid=stMain],section.main,div[data-testid=stAppViewContainer]')];
 const el = els.find(e => e.scrollHeight > e.clientHeight + 5) || document.scrollingElement;
 el.scrollTo({top: f * (el.scrollHeight - el.clientHeight), behavior: 'smooth'}); }"""

async def record(blocks, spans, out, args):
    from playwright.async_api import async_playwright
    size = args.size
    async with async_playwright() as pw:
        kw = {}
        import os
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and pathlib.Path("/opt/pw-browsers/chromium").exists():
            kw["executable_path"] = "/opt/pw-browsers/chromium"
        br = await pw.chromium.launch(**kw)
        ctx = await br.new_context(viewport={"width": size, "height": size}, record_video_dir=str(out / "raw"), record_video_size={"width": size, "height": size})
        t_ctx = time.time()
        page = await ctx.new_page()
        await page.goto(args.url); await page.wait_for_timeout(5000)
        if not args.keep_sidebar:
            await page.add_style_tag(content="[data-testid=stSidebar],[data-testid=stSidebarCollapsedControl],header[data-testid=stHeader]{display:none!important}")
        await page.evaluate(SUB_JS, "")
        await page.wait_for_timeout(int(LEAD * 1000))
        t0 = time.time(); offset = t0 - t_ctx
        async def until(sec):
            d = t0 + sec - time.time()
            if d > 0: await page.wait_for_timeout(int(d * 1000))
        k = 0
        for b in blocks:
            start = spans[k][0] if b["say"] else None
            if start is not None: await until(start)
            for a in b["do"]:
                cmd, _, arg = a.partition(" ")
                try:
                    if cmd == "scroll": await page.evaluate(SCROLL_JS, float(arg))
                    elif cmd == "click": await page.get_by_text(arg, exact=False).first.click(timeout=5000)
                    elif cmd == "wait": await page.wait_for_timeout(int(float(arg) * 1000))
                    else: print("unknown action:", a)
                except Exception as e:
                    print("action failed:", a, "-", str(e)[:80])
            for s in b["say"]:
                await until(spans[k][0]); await page.evaluate(SUB_JS, s); k += 1
                await until(spans[k - 1][1])
            await page.evaluate(SUB_JS, "")
        await until(spans[-1][1] + TAIL)
        await ctx.close(); await br.close()
    return offset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script"); ap.add_argument("out")
    ap.add_argument("--voice", default="en-US-AriaNeural"); ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--url", default="http://localhost:8501"); ap.add_argument("--keep-sidebar", action="store_true")
    ap.add_argument("--fake-voice", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blocks = parse(a.script); sentences = [s for b in blocks for s in b["say"]]
    spans = asyncio.run(make_audio(sentences, out, a.voice, a.fake_voice))
    print(f"narration: {spans[-1][1]:.0f} s, {len(sentences)} sentences")
    offset = asyncio.run(record(blocks, spans, out, a))
    stem = pathlib.Path(a.script).stem
    (out / f"{stem}.srt").write_text("".join(f"{i}\n{srt_time(s + offset)} --> {srt_time(e + offset)}\n{t}\n\n" for i, ((s, e), t) in enumerate(zip(spans, sentences), 1)), encoding="utf-8")
    vid = max((out / "raw").glob("*.webm"), key=lambda p: p.stat().st_mtime)
    ms = int(offset * 1000)
    r = run([FFMPEG, "-y", "-i", str(vid), "-i", str(out / "narration.mp3"), "-filter_complex", f"[1:a]adelay={ms}:all=1[a]",
             "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out / f"{stem}.mp4")])
    if r.returncode: print(r.stderr[-800:]); sys.exit(1)
    (out / "narration.mp3").replace(out / f"{stem}.mp3")
    for f in list(out.glob("s*.mp3")) + [out / "gap.mp3", out / "list.txt"]: f.unlink(missing_ok=True)
    print("done:", out / f"{stem}.mp4")

if __name__ == "__main__": main()
