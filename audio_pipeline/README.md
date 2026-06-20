# Timbre-space audio pipeline

Turns a **real** piece of music into the data behind `../timbre_space.html`:
real audio → **source separation** → per-stem **timbre features** → 3D point cloud + playable stems.

## What it does

1. **Loads a real clip** — by default the Spleeter demo song
   (`deezer/spleeter` `audio_example.mp3`, fetched from GitHub).
2. **Separates instruments** with **Spleeter `5stems`** (pretrained):
   `vocals · piano · drums · bass · other`.
   > Demucs gives cleaner stems, but its weights are only on hosts blocked by this
   > sandbox's network policy. Spleeter's weights are on GitHub, so it runs here.
   > **For higher-quality Demucs stems, run [`demucs_colab.ipynb`](demucs_colab.ipynb)
   > in Google Colab** (open network) — it exports the same `web/` files this script
   > does, so the visualization upgrades with no code changes.
3. **Extracts an MFCC timbre descriptor** per energetic frame of each stem
   (MFCC×20 + spectral centroid / bandwidth / contrast / rolloff + ZCR + flatness).
4. **Reduces all frames jointly to a 3D timbre map** with **t-SNE** (default) or
   PCA — so instruments *cluster on their own*, without the layout ever seeing the
   labels (cf. MFCC→t-SNE timbre space; Google "Infinite Drum Machine"). A
   silhouette score reports how well they separate. It also keeps a **raw
   feature-axis layout** (X = attack, Y = brightness, Z = spectral flux) so the
   viz can morph between "raw features" and the "learned t-SNE map".
5. **Exports** what the WebGL viz consumes:
   - `../web/timbre_data.json` — points (both layouts), per-stem centroids, order, silhouette
   - `../web/audio/<stem>.ogg` — mono stems, played back & cross-faded in the viz

Switch reducer via `REDUCER = "tsne"` / `"pca"` near the top of `separate_and_extract.py`.

## Run

```bash
pip install numpy soundfile librosa spleeter scikit-learn
python3 audio_pipeline/separate_and_extract.py
```

Then open the visualization (needs http, not `file://`, because it `fetch`es the data):

```bash
python3 -m http.server 8000
# visit http://localhost:8000/timbre_space.html
```

## Use your own song

Drop a file at `audio_pipeline/_work/example.mp3` (or edit `SAMPLE_MP3` / `SAMPLE_URL`
in `separate_and_extract.py`) and re-run. Stereo 44.1 kHz works best.
