#!/usr/bin/env python3
"""
Timbre-space pipeline: real audio -> source separation -> MFCC timbre features
-> dimensionality reduction (t-SNE / PCA) -> 3D timbre map.

This is the "professor-proof" version: point positions come from a real
**t-SNE (or PCA) of MFCC timbre features**, so instruments *cluster on their
own* (no hand-placed lanes) -- matching the standard timbre-space technique
(cf. MFCC + spectral-centroid -> t-SNE; Google "Infinite Drum Machine").

Pipeline
--------
1. Load a real multi-instrument clip.
2. Separate into stems with Spleeter (pretrained; Demucs weights are blocked by
   the sandbox network, see demucs_colab.ipynb for the higher-quality variant).
3. Per stem, per energetic frame, compute a timbre descriptor:
   MFCC(20) + spectral centroid/bandwidth/contrast/rolloff + ZCR + flatness.
4. Reduce ALL frames jointly to 3D with t-SNE (default) or PCA -> the timbre map.
   (Also keep a raw feature-axis layout: X=attack, Y=brightness, Z=flux, so the
    viz can morph between "raw features" and "learned t-SNE map".)
5. Export web/timbre_data.json + web/audio/<stem>.ogg.

Run:  python3 audio_pipeline/separate_and_extract.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import soundfile as sf
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK      = os.path.join(ROOT, "audio_pipeline", "_work")
WEB       = os.path.join(ROOT, "web")
WEB_AUDIO = os.path.join(WEB, "audio")
SAMPLE_MP3 = os.path.join(WORK, "example.mp3")
SAMPLE_URL = "https://raw.githubusercontent.com/deezer/spleeter/master/audio_example.mp3"
SAMPLE_NAME = "Spleeter demo clip (deezer/spleeter, audio_example.mp3)"

REDUCER = "tsne"          # "tsne" or "pca"
POINTS_PER_STEM = 600
SR = 22050

STEM_COLORS = {
    "vocals": "#ff5e9c", "piano": "#5be0ff", "drums": "#ffb347",
    "bass": "#9b7bff", "other": "#7cff8a", "guitar": "#ffd45e",
}
DEFAULT_COLOR = "#9fd0ff"


def ensure_sample():
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(SAMPLE_MP3):
        import urllib.request
        print("Downloading sample:", SAMPLE_URL)
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_MP3)
    wav, sr = sf.read(SAMPLE_MP3)
    return wav.astype("float32"), sr


def separate_spleeter(wav, sr):
    from spleeter.separator import Separator
    print("Separating with Spleeter 5stems (pretrained)...")
    sep = Separator("spleeter:5stems")
    stems = sep.separate(wav)
    out = {}
    for name, data in stems.items():
        mono = data.mean(axis=1).astype("float32")
        if sr != SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
        out[name] = mono
    return out


def stem_frames(y):
    """Per energetic frame: timbre descriptor + (attack, brightness, flux, rms)."""
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    mfcc     = librosa.feature.mfcc(S=librosa.power_to_db(S**2), sr=SR, n_mfcc=20)
    cent     = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bw       = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR)
    rolloff  = librosa.feature.spectral_rolloff(S=S, sr=SR)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    flat     = librosa.feature.spectral_flatness(S=S)[0]
    flux     = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=SR)
    rms      = librosa.feature.rms(S=S)[0]

    T = min(mfcc.shape[1], len(cent), len(bw), contrast.shape[1], len(rolloff), len(zcr), len(flat), len(flux), len(rms))
    feat = np.vstack([mfcc[:, :T], cent[:T], bw[:T], contrast[:, :T], rolloff[:T], zcr[:T], flat[:T]]).T  # (T, D)
    cent, flux, rms = cent[:T], flux[:T], rms[:T]

    # attack time per frame, from nearest onset
    onsets = list(librosa.onset.onset_detect(onset_envelope=flux, sr=SR, hop_length=hop, backtrack=True)) + [T]
    attack = np.zeros(T, "float32")
    for i in range(len(onsets) - 1):
        a, b = onsets[i], min(onsets[i] + 20, onsets[i + 1])
        if b > a:
            peak = a + int(np.argmax(rms[a:b]))
            attack[a:onsets[i + 1]] = np.clip((peak - a) * hop / SR * 1000.0, 1.0, 160.0)

    keep = np.where(rms > max(rms.mean() * 0.6, 1e-4))[0]
    if len(keep) == 0:
        keep = np.arange(T)
    if len(keep) > POINTS_PER_STEM:
        keep = np.random.choice(keep, POINTS_PER_STEM, replace=False)
    return feat[keep], attack[keep], cent[keep], flux[keep], rms[keep]


def axis_scale(a, span, lo_p=2, hi_p=98):
    lo, hi = np.percentile(a, lo_p), np.percentile(a, hi_p)
    rng = hi - lo if hi > lo else 1.0
    return (np.clip((a - lo) / rng, 0, 1) - 0.5) * 2 * span


def main():
    np.random.seed(7)
    os.makedirs(WEB_AUDIO, exist_ok=True)
    wav, sr = ensure_sample()
    print("Input clip:", wav.shape, sr, "Hz", round(len(wav) / sr, 1), "s")

    stems = separate_spleeter(wav, sr)

    names, feats, atk, brt, flx, rms_, label = [], [], [], [], [], [], []
    for si, (name, y) in enumerate(stems.items()):
        f, a, c, fl, r = stem_frames(y)
        names.append(name)
        feats.append(f); atk.append(a); brt.append(c); flx.append(fl); rms_.append(r)
        label.append(np.full(len(a), si))
        sf.write(os.path.join(WEB_AUDIO, f"{name}.ogg"), y, SR, format="OGG", subtype="VORBIS")
        print(f"  {name:7s}: {len(a)} frames  attack~{np.median(a):.0f}ms bright~{np.median(c):.0f}Hz")

    X      = np.vstack(feats)
    labels = np.concatenate(label)
    atk    = np.concatenate(atk); brt = np.concatenate(brt)
    flx    = np.concatenate(flx); rms_ = np.concatenate(rms_)
    print(f"\nTimbre feature matrix: {X.shape}  ({X.shape[1]} dims/frame)")

    # --- dimensionality reduction to 3D (the timbre map) ---
    Xs = StandardScaler().fit_transform(X)
    if REDUCER == "tsne":
        Xp = PCA(n_components=min(20, Xs.shape[1]), random_state=7).fit_transform(Xs)
        emb = TSNE(n_components=3, perplexity=30, init="pca",
                   learning_rate="auto", random_state=7).fit_transform(Xp)
        method = "Spleeter stems + 3D t-SNE of MFCC timbre features"
    else:
        emb = PCA(n_components=3, random_state=7).fit_transform(Xs)
        method = "Spleeter stems + 3D PCA of MFCC timbre features"

    # how well do instruments separate in the map? (report a real number)
    try:
        sil = silhouette_score(emb, labels)
        print(f"{REDUCER.upper()} map silhouette (instrument separation): {sil:.3f}")
    except Exception:
        sil = None

    # scale embedding axes to world space
    tx = axis_scale(emb[:, 0], 30); ty = axis_scale(emb[:, 1], 20); tz = axis_scale(emb[:, 2], 30)
    # raw feature-axis layout (X=attack, Y=brightness, Z=flux) for the morph view
    fx = axis_scale(atk, 28); fy = axis_scale(np.log1p(brt), 18); fz = axis_scale(flx, 28)
    bnorm = (np.log1p(brt) - np.log1p(brt).min()) / (np.ptp(np.log1p(brt)) + 1e-9)

    stems_meta, points = [], []
    for si, name in enumerate(names):
        m = labels == si
        stems_meta.append({
            "name": name, "color": STEM_COLORS.get(name, DEFAULT_COLOR),
            "centroidTsne": [float(tx[m].mean()), float(ty[m].mean()), float(tz[m].mean())],
            "centroidFeat": [float(fx[m].mean()), float(fy[m].mean()), float(fz[m].mean())],
            "count": int(m.sum()),
            "meanBrightnessHz": float(np.median(brt[m])),
            "audio": f"web/audio/{name}.ogg",
        })
    rmsn = rms_ / (rms_.max() + 1e-9)
    for i in range(len(tx)):
        points.append([round(float(tx[i]),1), round(float(ty[i]),1), round(float(tz[i]),1),
                       round(float(fx[i]),1), round(float(fy[i]),1), round(float(fz[i]),1),
                       int(labels[i]), round(float(rmsn[i]),2), round(float(bnorm[i]),2)])

    order = sorted(range(len(stems_meta)), key=lambda i: stems_meta[i]["meanBrightnessHz"])
    data = {
        "sample": SAMPLE_NAME, "method": method,
        "reducer": REDUCER,
        "silhouette": (round(float(sil), 3) if sil is not None else None),
        "axesTsne": "abstract t-SNE dims · proximity = timbral similarity · colour = instrument",
        "axesFeat": {"x": "Attack Time", "y": "Brightness (spectral centroid)", "z": "Spectral Flux"},
        "stems": stems_meta, "trajectory": order, "points": points,
        "pointFormat": ["tx","ty","tz","fx","fy","fz","stemIndex","rms","brightness"],
    }
    with open(os.path.join(WEB, "timbre_data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"\nWrote web/timbre_data.json  ({len(points)} points, {len(stems_meta)} stems, reducer={REDUCER})")


if __name__ == "__main__":
    main()
