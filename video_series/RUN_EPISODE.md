# Make one episode (on the laptop)

Once: `pip install edge-tts playwright imageio-ffmpeg` then `playwright install chromium ffmpeg`

1. Start the dashboard (`streamlit run dashboard/app.py`), open Overview and check the date and the green status line.
2. From the repo folder:
```
python video_series\tools\make_episode.py video_series\scripts\ep01_the_plant_in_one_minute.md "D:\Post Doc Project\Episodes\Episode 01 - The plant in one minute" --voice en-US-AriaNeural
```
Result in that folder: ep01_the_plant_in_one_minute.mp4 (square, subtitles, AI voice), .srt, .mp3.

- `--keep-sidebar` shows the sidebar. `--size 1080` is the square size.
- Voices: en-US-AriaNeural, en-GB-SoniaNeural, en-IN-NeerjaNeural, en-US-GuyNeural, en-GB-RyanNeural.
- In a script, `Do: scroll 0.3 | click <button text> | wait 2` runs at the start of a block; each spoken line is one subtitle.
