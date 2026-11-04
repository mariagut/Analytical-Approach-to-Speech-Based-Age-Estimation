"""
extract_n.py
============
Rozszerzenie extract_133.py do **464 cech głosu** — clinical-grade + paper AUX (277).

Funkcje publiczne:
  extract_n(path)             → dict z 464 cechami (single-shot, całe nagranie)
  extract_n_sliding(path, ..) → list[dict] (sliding window, 464 cech / okno)

Grupy (cumulative):
  feats      (100): jitter(5) + shimmer(6) + HNR(2) + formanty(9) + MFCC×13(78)
  direct     ( 25): env(2) + mod(2) + rms/centroid/bandwidth/rolloff/zcr × 4stat + aci
  feats_4    (  8): sylaby(4) + repetition(2) + spec_entropy(2)
  nowe       ( 17): F0(7) + formant_bandwidths(6) + CPP(2) + spectral_tilt(2)
  ──── EXT (37) ───────────────────────────────────────────────────────────────
  Tier-1 (17): BER(5) + Hammarberg(1) + loudness_dB(3) + H1H2/H1A1/H1A3(6) + flatness(2)
  Tier-2 (11): CPPS(2) + GNE(2) + F0 contour(3) + pause/rate(4)
  Tier-3 ( 9): DFA(1) + PPE(1) + SHR(2) + LPC residual(3) + SPI(1) + VTI(1)
  ──── AUX (277) — paper-style mean/std/var ───────────────────────────────────
  MFCC      (180): MFCC×20 + MFCC∆×20 + MFCC∆∆×20  (każde mean/std/var = 60)
  Chroma CENS(36): 12 binów × {mean,std,var}
  Tonnetz   ( 18): 6 dim × {mean,std,var}  (paper 7, librosa 6)
  Spec.contr( 24): 8 pasm × {mean,std,var}  (librosa n_bands=7)
  Scalary   ( 18): rms, spec_centroid/bw/rolloff/flatness, zcr × {mean,std,var}
  Tempo     (  1): BPM (skalar, bez stats)

Zależności: librosa, parselmouth, scipy, numpy
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import scipy.signal as ssignal
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from scipy.stats import entropy
from scipy.ndimage import uniform_filter

# ── stałe ────────────────────────────────────────────────────────────────────
SR           = 16000
N_FFT        = 1024
HOP          = 256
N_MFCC       = 13
F0_MIN       = 75
F0_MAX       = 600
MAX_FORMANTS = 5
MAX_FREQ     = 5500

# Stałe dla EXT (Tier 1–3)
BAND_EDGES   = [0, 500, 1000, 2000, 4000, 8000]   # Hz, eGeMAPS-style
N_FFT_FINE   = 2048   # wyższa rozdzielczość dla SHR / H1-H2 / harmonicznych


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ŁADOWANIE
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    return y, snd, S


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATS (100) — jitter / shimmer / HNR / formanty / MFCC
# ═══════════════════════════════════════════════════════════════════════════════

def _jitter(snd):
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
        return dict(
            jitter_local     = call(pp, "Get jitter (local)",          0, 0, 0.0001, 0.02, 1.3),
            jitter_local_abs = call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3),
            jitter_rap       = call(pp, "Get jitter (rap)",            0, 0, 0.0001, 0.02, 1.3),
            jitter_ppq5      = call(pp, "Get jitter (ppq5)",           0, 0, 0.0001, 0.02, 1.3),
            jitter_ddp       = call(pp, "Get jitter (ddp)",            0, 0, 0.0001, 0.02, 1.3),
        )
    except Exception:
        return {k: np.nan for k in
                ["jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp"]}


def _shimmer(snd):
    _A = (0, 0, 0.0001, 0.02, 1.3, 1.6)
    try:
        pp  = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
        loc = call([snd, pp], "Get shimmer (local)",  *_A)
        try:
            loc_db = call([snd, pp], "Get shimmer (local, dB)", *_A)
        except Exception:
            loc_db = float(20 * np.log10(1 + loc)) if np.isfinite(loc) else np.nan
        return dict(
            shimmer_local    = loc,
            shimmer_apq3     = call([snd, pp], "Get shimmer (apq3)",  *_A),
            shimmer_apq5     = call([snd, pp], "Get shimmer (apq5)",  *_A),
            shimmer_apq11    = call([snd, pp], "Get shimmer (apq11)", *_A),
            shimmer_dda      = call([snd, pp], "Get shimmer (dda)",   *_A),
            shimmer_local_dB = loc_db,
        )
    except Exception:
        return {k: np.nan for k in
                ["shimmer_local", "shimmer_local_dB", "shimmer_apq3",
                 "shimmer_apq5", "shimmer_apq11", "shimmer_dda"]}


def _hnr(snd):
    try:
        h = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
        return dict(
            hnr_mean = call(h, "Get mean",               0, 0),
            hnr_std  = call(h, "Get standard deviation", 0, 0),
        )
    except Exception:
        return dict(hnr_mean=np.nan, hnr_std=np.nan)


def _formants(snd):
    try:
        f = snd.to_formant_burg(max_number_of_formants=MAX_FORMANTS,
                                maximum_formant=MAX_FREQ)
        vals = {n: [] for n in range(1, 4)}
        for t in f.xs():
            for n in range(1, 4):
                v = f.get_value_at_time(n, t)
                if np.isfinite(v) and v > 0:
                    vals[n].append(v)
        out = {}
        for n in range(1, 4):
            a = np.array(vals[n])
            k = f"F{n}"
            out[f"{k}_mean"]   = float(np.mean(a))   if len(a) else np.nan
            out[f"{k}_std"]    = float(np.std(a))    if len(a) else np.nan
            out[f"{k}_median"] = float(np.median(a)) if len(a) else np.nan
        return out
    except Exception:
        return {f"F{n}_{s}": np.nan
                for n in range(1, 4) for s in ("mean", "std", "median")}


def _mfcc(y):
    mfcc  = librosa.feature.mfcc(y=y.astype(np.float32), sr=SR, n_mfcc=N_MFCC)
    d     = librosa.feature.delta(mfcc)
    dd    = librosa.feature.delta(mfcc, order=2)
    out   = {}
    for i in range(N_MFCC):
        p = f"mfcc{i+1:02d}"
        out[f"{p}_mean"]    = float(np.mean(mfcc[i]))
        out[f"{p}_std"]     = float(np.std(mfcc[i]))
        out[f"d_{p}_mean"]  = float(np.mean(d[i]))
        out[f"d_{p}_std"]   = float(np.std(d[i]))
        out[f"dd_{p}_mean"] = float(np.mean(dd[i]))
        out[f"dd_{p}_std"]  = float(np.std(dd[i]))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DIRECT (25) — env / mod / frame variability / aci
# ═══════════════════════════════════════════════════════════════════════════════

def _direct(y, S):
    rms       = librosa.feature.rms(y=y, hop_length=HOP)[0]
    centroid  = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    rolloff   = librosa.feature.spectral_rolloff(S=S, sr=SR)[0]
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]

    def _var4(v, name):
        dx = np.diff(v)
        cv = float(np.std(v) / (np.abs(np.mean(v)) + 1e-9))
        return {
            f"{name}_mean":      float(np.mean(v)),
            f"{name}_std":       float(np.std(v)),
            f"{name}_cv":        cv,
            f"{name}_delta_std": float(np.std(dx)),
        }

    out = {}
    out.update(_var4(rms,       "rms"))
    out.update(_var4(centroid,  "centroid"))
    out.update(_var4(bandwidth, "bandwidth"))
    out.update(_var4(rolloff,   "rolloff"))
    out.update(_var4(zcr,       "zcr"))

    env    = np.abs(ssignal.hilbert(y))
    env_ds = ssignal.resample(env, max(1, len(env) // HOP))
    out["env_mean"] = float(np.mean(env_ds))
    out["env_std"]  = float(np.std(env_ds))

    mod_spec        = np.abs(np.fft.rfft(rms))
    out["mod_std"]  = float(np.std(mod_spec))
    out["mod_entropy"] = float(entropy(mod_spec / (np.sum(mod_spec) + 1e-9)))

    diff = np.abs(np.diff(S, axis=1))
    out["aci"] = float(np.sum(diff) / (np.sum(S) + 1e-9))

    return out, rms


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NOWE (17) — F0 / formant bandwidths / CPP / spectral tilt
# ═══════════════════════════════════════════════════════════════════════════════

def _f0(snd):
    """F0 mean/std/min/max/range, voicing_ratio, voice_breaks — 7 cech."""
    try:
        pitch   = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr  = pitch.selected_array["frequency"]
        voiced  = f0_arr[f0_arr > 0]
        n_total = len(f0_arr)

        is_voiced    = (f0_arr > 0).astype(int)
        voice_breaks = int(np.sum(np.diff(is_voiced) == -1))

        if len(voiced) == 0:
            return {k: np.nan for k in
                    ["f0_mean","f0_std","f0_min","f0_max","f0_range",
                     "voicing_ratio","voice_breaks"]}
        return dict(
            f0_mean       = float(np.mean(voiced)),
            f0_std        = float(np.std(voiced)),
            f0_min        = float(np.min(voiced)),
            f0_max        = float(np.max(voiced)),
            f0_range      = float(np.max(voiced) - np.min(voiced)),
            voicing_ratio = float(len(voiced) / max(n_total, 1)),
            voice_breaks  = float(voice_breaks),
        )
    except Exception:
        return {k: np.nan for k in
                ["f0_mean","f0_std","f0_min","f0_max","f0_range",
                 "voicing_ratio","voice_breaks"]}


def _formant_bandwidths(snd):
    """Szerokości formantów B1/B2/B3 (mean, std) — 6 cech."""
    try:
        f  = snd.to_formant_burg(max_number_of_formants=MAX_FORMANTS,
                                  maximum_formant=MAX_FREQ)
        bv = {n: [] for n in range(1, 4)}
        for t in f.xs():
            for n in range(1, 4):
                b = call(f, "Get bandwidth at time", n, t, "Hertz", "Linear")
                if np.isfinite(b) and b > 0:
                    bv[n].append(b)
        out = {}
        for n in range(1, 4):
            a = np.array(bv[n])
            out[f"B{n}_mean"] = float(np.mean(a)) if len(a) else np.nan
            out[f"B{n}_std"]  = float(np.std(a))  if len(a) else np.nan
        return out
    except Exception:
        return {f"B{n}_{s}": np.nan for n in range(1, 4) for s in ("mean", "std")}


def _cpp(y):
    """Cepstral Peak Prominence (mean, std po ramkach) — 2 cechy."""
    try:
        frame_len = int(0.025 * SR)
        hop_len   = int(0.010 * SR)
        q_min     = max(1, int(SR / F0_MAX))
        q_max     = min(frame_len // 2 - 1, int(SR / F0_MIN))

        frames    = librosa.util.frame(y, frame_length=frame_len,
                                       hop_length=hop_len, axis=0)
        win       = np.hanning(frame_len)
        cpp_vals  = []
        for frame in frames:
            S_log    = np.log(np.abs(np.fft.rfft(frame * win)) ** 2 + 1e-9)
            cep      = np.abs(np.fft.irfft(S_log))[:frame_len // 2]
            peak_i   = q_min + np.argmax(cep[q_min:q_max + 1])
            peak_v   = cep[peak_i]
            x        = np.arange(len(cep))
            slope, intercept = np.polyfit(x, cep, 1)
            cpp_vals.append(peak_v - (slope * peak_i + intercept))

        if cpp_vals:
            return dict(cpp_mean=float(np.mean(cpp_vals)),
                        cpp_std =float(np.std(cpp_vals)))
        return dict(cpp_mean=np.nan, cpp_std=np.nan)
    except Exception:
        return dict(cpp_mean=np.nan, cpp_std=np.nan)


def _spectral_tilt(S):
    """Nachylenie obwiedni widmowej i alpha ratio — 2 cechy."""
    freqs    = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mag_mean = np.mean(S, axis=1)

    mask     = freqs > 0
    slope, _ = np.polyfit(np.log(freqs[mask] + 1e-9),
                           np.log(mag_mean[mask] + 1e-9), 1)

    idx_1k   = int(np.searchsorted(freqs, 1000))
    alpha    = float(np.sum(mag_mean[:idx_1k]) /
                     (np.sum(mag_mean[idx_1k:]) + 1e-9))

    return dict(spectral_slope=float(slope), alpha_ratio=alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEATS_4 (8) — sylaby / repetition / spec_entropy
# ═══════════════════════════════════════════════════════════════════════════════

def _syllables(y):
    env = np.abs(ssignal.hilbert(y))
    env_smooth = ssignal.medfilt(env, kernel_size=7)
    threshold  = np.mean(env_smooth) * 1.2
    active     = env_smooth > threshold

    segments, start = [], None
    for i, val in enumerate(active):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(env_smooth)))

    if len(segments) < 2:
        return dict(syllable_count=len(segments),
                    syllable_duration_std=0.0,
                    syllable_interval_std=0.0,
                    syllable_rate=float(len(segments)))

    durations  = [(e - s) / SR for s, e in segments]
    intervals  = [(segments[i+1][0] - segments[i][1]) / SR
                  for i in range(len(segments) - 1)]
    return dict(
        syllable_count        = len(segments),
        syllable_duration_std = float(np.std(durations)),
        syllable_interval_std = float(np.std(intervals)),
        syllable_rate         = float(len(segments) / (segments[-1][1] / SR)),
    )


def _repetition(rms):
    acf  = np.correlate(rms, rms, mode='full')
    acf  = acf[len(acf) // 2:]
    peaks, _ = find_peaks(acf, height=np.max(acf) * 0.3)
    return dict(
        repetition_peak_count    = int(len(peaks)),
        repetition_strength_std  = float(np.std(acf[peaks])) if len(peaks) > 0 else 0.0,
    )


def _spec_entropy_variability(S):
    ps         = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
    spec_entr  = entropy(ps, axis=0)
    dx         = np.diff(spec_entr)
    return dict(
        spec_entropy_std       = float(np.std(spec_entr)),
        spec_entropy_delta_std = float(np.std(dx)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXT TIER-1 (17) — BER / Hammarberg / loudness / H1-H2-A1-A3 / flatness
# ═══════════════════════════════════════════════════════════════════════════════

def _band_energy_ratios(S):
    """5 band energy ratios (eGeMAPS edges) na uśrednionym widmie magnitudy."""
    freqs    = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mag_mean = np.mean(S, axis=1)
    total    = float(np.sum(mag_mean) + 1e-9)
    out = {}
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        idx = (freqs >= lo) & (freqs < hi)
        out[f"ber_{lo}_{hi}"] = float(np.sum(mag_mean[idx]) / total)
    return out


def _hammarberg(S):
    """Hammarberg index (dB): max{0–2 kHz} / max{2–5 kHz} — biomarker chrypki."""
    freqs    = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mag_mean = np.mean(S, axis=1)
    idx_2k   = int(np.searchsorted(freqs, 2000))
    idx_5k   = int(np.searchsorted(freqs, 5000))
    low_max  = float(np.max(mag_mean[:idx_2k]))      if idx_2k > 0       else 0.0
    high_max = float(np.max(mag_mean[idx_2k:idx_5k])) if idx_5k > idx_2k else 1e-9
    return dict(hammarberg_db=float(20.0 * np.log10((low_max + 1e-9) / (high_max + 1e-9))))


def _loudness_db(y):
    """RMS-loudness w dB: mean / std / range (P95–P5) — 3 cechy."""
    rms_frames = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = 20.0 * np.log10(rms_frames + 1e-9)
    return dict(
        loudness_db_mean  = float(np.mean(rms_db)),
        loudness_db_std   = float(np.std(rms_db)),
        loudness_db_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)),
    )


def _glottal_source(y, snd):
    """H1-H2 / H1-A1 / H1-A3 (mean, std po voiced ramkach) — 6 cech.
    H1=ampl. F0, H2=ampl. 2·F0, A1/A3=ampl. najbliższego binu F1/F3 (dB)."""
    keys_nan = {f"{k}_{s}": np.nan
                for k in ("h1h2","h1a1","h1a3") for s in ("mean","std")}
    try:
        pitch   = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr  = pitch.selected_array["frequency"]
        p_times = pitch.xs()

        formant = snd.to_formant_burg(max_number_of_formants=MAX_FORMANTS,
                                       maximum_formant=MAX_FREQ)

        Sf = np.abs(librosa.stft(y.astype(np.float32),
                                  n_fft=N_FFT_FINE, hop_length=HOP))
        S_db   = 20.0 * np.log10(Sf + 1e-9)
        S_freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT_FINE)
        S_times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                          sr=SR, hop_length=HOP)

        def _db_at(spec, f):
            if f <= 0 or f >= SR / 2:
                return None
            return float(spec[int(np.argmin(np.abs(S_freqs - f)))])

        h1h2, h1a1, h1a3 = [], [], []
        for t, f0 in zip(p_times, f0_arr):
            if f0 <= 0:
                continue
            f_idx = int(np.argmin(np.abs(S_times - t)))
            spec = S_db[:, f_idx]
            f1 = formant.get_value_at_time(1, t)
            f3 = formant.get_value_at_time(3, t)
            h1 = _db_at(spec, f0)
            h2 = _db_at(spec, 2 * f0)
            a1 = _db_at(spec, f1) if np.isfinite(f1) else None
            a3 = _db_at(spec, f3) if np.isfinite(f3) else None
            if h1 is not None and h2 is not None: h1h2.append(h1 - h2)
            if h1 is not None and a1 is not None: h1a1.append(h1 - a1)
            if h1 is not None and a3 is not None: h1a3.append(h1 - a3)

        def _stats(vals):
            if not vals:
                return np.nan, np.nan
            return float(np.mean(vals)), float(np.std(vals))

        h1h2_m, h1h2_s = _stats(h1h2)
        h1a1_m, h1a1_s = _stats(h1a1)
        h1a3_m, h1a3_s = _stats(h1a3)
        return dict(h1h2_mean=h1h2_m, h1h2_std=h1h2_s,
                    h1a1_mean=h1a1_m, h1a1_std=h1a1_s,
                    h1a3_mean=h1a3_m, h1a3_std=h1a3_s)
    except Exception:
        return keys_nan


def _spectral_flatness(y):
    """Wiener entropy (spectral flatness) — 2 cechy (mean, std)."""
    try:
        flat = librosa.feature.spectral_flatness(y=y.astype(np.float32),
                                                  n_fft=N_FFT, hop_length=HOP)[0]
        return dict(flatness_mean=float(np.mean(flat)),
                    flatness_std =float(np.std(flat)))
    except Exception:
        return dict(flatness_mean=np.nan, flatness_std=np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXT TIER-2 (11) — CPPS / GNE / F0 contour / pause-rate
# ═══════════════════════════════════════════════════════════════════════════════

def _cpps(y):
    """CPP Smoothed (Hillenbrand): 2D moving average na cepstrogramie — 2 cechy."""
    try:
        frame_len = int(0.025 * SR)
        hop_len   = int(0.010 * SR)
        q_min = max(1, int(SR / F0_MAX))
        q_max = min(frame_len // 2 - 1, int(SR / F0_MIN))

        frames = librosa.util.frame(y, frame_length=frame_len,
                                     hop_length=hop_len, axis=0)
        win    = np.hanning(frame_len)

        ceps = []
        for fr in frames:
            S_log = np.log(np.abs(np.fft.rfft(fr * win)) ** 2 + 1e-9)
            cep   = np.abs(np.fft.irfft(S_log))[:frame_len // 2]
            ceps.append(cep)
        if not ceps:
            return dict(cpps_mean=np.nan, cpps_std=np.nan)
        ceps = np.asarray(ceps)

        # Hillenbrand smoothing: ~10 ms in time × ~1 ms in quefrency
        ceps_s = uniform_filter(ceps, size=(3, 11), mode="nearest")

        cpps_vals = []
        for cep in ceps_s:
            peak_i = q_min + int(np.argmax(cep[q_min:q_max + 1]))
            peak_v = float(cep[peak_i])
            x = np.arange(len(cep))
            slope, intercept = np.polyfit(x, cep, 1)
            cpps_vals.append(peak_v - (slope * peak_i + intercept))

        return dict(cpps_mean=float(np.mean(cpps_vals)),
                    cpps_std =float(np.std(cpps_vals)))
    except Exception:
        return dict(cpps_mean=np.nan, cpps_std=np.nan)


def _gne(y):
    """Glottal-to-Noise Excitation (Michaelis 1997) — 2 cechy.
    Bandpass → Hilbert envelope → max cross-correlation across band pairs."""
    try:
        bands = [(500, 1500), (1000, 2000), (1500, 2500), (2000, 3000)]
        seg_len = SR  # 1 s segments
        n_segs  = max(1, len(y) // seg_len)
        gne_vals = []
        for s in range(n_segs):
            y_seg = y[s * seg_len:(s + 1) * seg_len]
            if len(y_seg) < SR // 2:
                continue
            envs = []
            for lo, hi in bands:
                b, a = butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band")
                yf = filtfilt(b, a, y_seg)
                envs.append(np.abs(hilbert(yf)))
            cors = []
            for i in range(len(envs)):
                for j in range(i + 1, len(envs)):
                    e1 = envs[i] - np.mean(envs[i])
                    e2 = envs[j] - np.mean(envs[j])
                    n  = np.sqrt(np.sum(e1 ** 2) * np.sum(e2 ** 2))
                    if n > 0:
                        c = float(np.max(np.correlate(e1, e2, mode="full")) / n)
                        cors.append(c)
            if cors:
                gne_vals.append(np.max(cors))
        if not gne_vals:
            return dict(gne_mean=np.nan, gne_std=np.nan)
        return dict(gne_mean=float(np.mean(gne_vals)),
                    gne_std =float(np.std(gne_vals)))
    except Exception:
        return dict(gne_mean=np.nan, gne_std=np.nan)


def _f0_contour(snd):
    """F0 slope per voiced segment + trajectory entropy — 3 cechy."""
    keys_nan = dict(f0_slope_mean=np.nan, f0_slope_std=np.nan,
                    f0_traj_entropy=np.nan)
    try:
        pitch  = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr = pitch.selected_array["frequency"]
        times  = pitch.xs()
        if len(times) < 2:
            return keys_nan
        dt = float(times[1] - times[0])

        is_v = f0_arr > 0
        slopes = []
        i = 0
        while i < len(is_v):
            if is_v[i]:
                j = i
                while j < len(is_v) and is_v[j]:
                    j += 1
                seg = f0_arr[i:j]
                if len(seg) >= 5:
                    x = np.arange(len(seg)) * dt
                    m, _ = np.polyfit(x, seg, 1)
                    slopes.append(float(m))
                i = j
            else:
                i += 1

        voiced = f0_arr[is_v]
        if len(voiced) >= 10:
            df0 = np.diff(voiced)
            hist, _ = np.histogram(df0, bins=20, density=True)
            traj_ent = float(entropy(hist + 1e-9))
        else:
            traj_ent = np.nan

        return dict(
            f0_slope_mean   = float(np.mean(slopes)) if slopes else np.nan,
            f0_slope_std    = float(np.std(slopes))  if slopes else np.nan,
            f0_traj_entropy = traj_ent,
        )
    except Exception:
        return keys_nan


def _pause_rate(y, snd):
    """Pause count, mean pause dur, articulation rate, voiced ratio — 4 cechy."""
    try:
        pitch  = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr = pitch.selected_array["frequency"]
        times  = pitch.xs()
        if len(times) < 2:
            return dict(pause_count=0, pause_mean_dur=0.0,
                        articulation_rate=np.nan, voiced_ratio=np.nan)
        dt = float(times[1] - times[0])
        is_v = f0_arr > 0

        min_pause_frames = max(1, int(0.2 / dt))   # próg 200 ms
        pauses = []
        i = 0
        while i < len(is_v):
            if not is_v[i]:
                j = i
                while j < len(is_v) and not is_v[j]:
                    j += 1
                if (j - i) >= min_pause_frames:
                    pauses.append((j - i) * dt)
                i = j
            else:
                i += 1

        total_dur     = len(y) / SR
        voiced_ratio  = float(np.sum(is_v)) / max(1, len(is_v))
        # articulation_rate: voiced frames / total non-pause time
        non_pause_dur = max(1e-9, total_dur - sum(pauses))
        artic_rate    = float((np.sum(is_v) * dt) / non_pause_dur)

        return dict(
            pause_count       = int(len(pauses)),
            pause_mean_dur    = float(np.mean(pauses)) if pauses else 0.0,
            articulation_rate = artic_rate,
            voiced_ratio      = float(voiced_ratio),
        )
    except Exception:
        return dict(pause_count=0, pause_mean_dur=0.0,
                    articulation_rate=np.nan, voiced_ratio=np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXT TIER-3 (9) — DFA / PPE / SHR / LPC residual / SPI / VTI
# ═══════════════════════════════════════════════════════════════════════════════

def _dfa(snd):
    """Detrended Fluctuation Analysis na konturze F0 — 1 cecha (α)."""
    try:
        pitch  = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr = pitch.selected_array["frequency"]
        voiced = f0_arr[f0_arr > 0]
        if len(voiced) < 50:
            return dict(dfa_alpha=np.nan)

        x = np.cumsum(voiced - np.mean(voiced))
        scales = np.unique(np.logspace(1, np.log10(max(10, len(x) // 4)), 15).astype(int))
        F = []
        used = []
        for s in scales:
            n_segs = len(x) // s
            if n_segs < 2:
                continue
            rms = []
            for i in range(n_segs):
                seg = x[i * s:(i + 1) * s]
                t   = np.arange(s)
                p   = np.polyfit(t, seg, 1)
                detr = seg - (p[0] * t + p[1])
                rms.append(np.sqrt(np.mean(detr ** 2)))
            F.append(np.mean(rms))
            used.append(s)
        if len(F) < 3:
            return dict(dfa_alpha=np.nan)
        log_s = np.log(np.asarray(used))
        log_F = np.log(np.asarray(F) + 1e-9)
        alpha, _ = np.polyfit(log_s, log_F, 1)
        return dict(dfa_alpha=float(alpha))
    except Exception:
        return dict(dfa_alpha=np.nan)


def _ppe(snd):
    """Pitch Period Entropy — 1 cecha (znormalizowana entropia residuum log-F0)."""
    try:
        pitch  = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr = pitch.selected_array["frequency"]
        voiced = f0_arr[f0_arr > 0]
        if len(voiced) < 30:
            return dict(ppe=np.nan)
        log_f0 = np.log(voiced)
        # whitening: AR(1) prediction residual
        x  = log_f0[1:]
        x0 = log_f0[:-1]
        rho = float(np.corrcoef(x, x0)[0, 1]) if len(x) > 1 else 0.0
        res = x - rho * x0
        n_bins = 30
        hist, _ = np.histogram(res, bins=n_bins, density=True)
        hist = hist + 1e-9
        h = float(entropy(hist) / np.log(n_bins))
        return dict(ppe=h)
    except Exception:
        return dict(ppe=np.nan)


def _shr(y, snd):
    """Subharmonic-to-Harmonic Ratio — 2 cechy (mean, std po voiced ramkach)."""
    try:
        pitch  = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_arr = pitch.selected_array["frequency"]
        p_times = pitch.xs()
        if len(p_times) < 2:
            return dict(shr_mean=np.nan, shr_std=np.nan)

        Sf = np.abs(librosa.stft(y.astype(np.float32),
                                  n_fft=N_FFT_FINE, hop_length=HOP))
        S_freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT_FINE)
        S_times = librosa.frames_to_time(np.arange(Sf.shape[1]),
                                          sr=SR, hop_length=HOP)

        shrs = []
        for t, f0 in zip(p_times, f0_arr):
            if f0 <= 0 or 2 * f0 >= SR / 2:
                continue
            f_idx = int(np.argmin(np.abs(S_times - t)))
            spec = Sf[:, f_idx]
            even_amp = 0.0
            odd_amp  = 0.0
            for k in range(1, 6):
                fh = k * f0
                if fh >= SR / 2:
                    break
                even_amp += float(spec[int(np.argmin(np.abs(S_freqs - fh)))])
                fsh = (k - 0.5) * f0
                if 0 < fsh < SR / 2:
                    odd_amp += float(spec[int(np.argmin(np.abs(S_freqs - fsh)))])
            if even_amp > 0:
                shrs.append(odd_amp / (even_amp + 1e-9))
        if not shrs:
            return dict(shr_mean=np.nan, shr_std=np.nan)
        return dict(shr_mean=float(np.mean(shrs)),
                    shr_std =float(np.std(shrs)))
    except Exception:
        return dict(shr_mean=np.nan, shr_std=np.nan)


def _lpc_residual(y):
    """LPC(16) residual energy + residual periodicity (HNR-like) — 3 cechy."""
    try:
        order     = 16
        frame_len = int(0.025 * SR)
        hop_len   = int(0.010 * SR)
        frames = librosa.util.frame(y, frame_length=frame_len,
                                     hop_length=hop_len, axis=0)
        win    = np.hanning(frame_len)

        rms_vals = []
        hnr_vals = []
        min_lag  = int(SR / F0_MAX)
        max_lag  = int(SR / F0_MIN)

        for fr in frames:
            fr_w = fr * win
            try:
                a = librosa.lpc(fr_w, order=order)
            except Exception:
                continue
            res = ssignal.lfilter(a, [1.0], fr_w)
            r0  = float(np.mean(res ** 2))
            if r0 <= 0:
                continue
            rms_vals.append(np.sqrt(r0))
            acf = np.correlate(res, res, mode="full")
            acf = acf[len(acf) // 2:]
            if max_lag > min_lag and max_lag < len(acf) and acf[0] > 0:
                peak = float(np.max(acf[min_lag:max_lag]))
                hnr_vals.append(peak / acf[0])

        return dict(
            lpc_res_rms_mean = float(np.mean(rms_vals)) if rms_vals else np.nan,
            lpc_res_rms_std  = float(np.std(rms_vals))  if rms_vals else np.nan,
            lpc_res_hnr_mean = float(np.mean(hnr_vals)) if hnr_vals else np.nan,
        )
    except Exception:
        return dict(lpc_res_rms_mean=np.nan, lpc_res_rms_std=np.nan,
                    lpc_res_hnr_mean=np.nan)


def _spi(S):
    """Soft Phonation Index: energia 70–1600 Hz / energia 1600–4500 Hz — 1 cecha."""
    try:
        freqs    = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        mag_mean = np.mean(S, axis=1)
        low      = (freqs >= 70)   & (freqs < 1600)
        high     = (freqs >= 1600) & (freqs < 4500)
        low_e    = float(np.sum(mag_mean[low]))
        high_e   = float(np.sum(mag_mean[high])) + 1e-9
        return dict(spi=float(low_e / high_e))
    except Exception:
        return dict(spi=np.nan)


def _vti(S):
    """Voice Turbulence Index: szum 2800–5800 Hz / sygnał 70–4500 Hz — 1 cecha."""
    try:
        freqs    = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        mag_mean = np.mean(S, axis=1)
        sig      = (freqs >= 70)   & (freqs < 4500)
        noise    = (freqs >= 2800) & (freqs < 5800)
        sig_e    = float(np.sum(mag_mean[sig])) + 1e-9
        noise_e  = float(np.sum(mag_mean[noise]))
        return dict(vti=float(noise_e / sig_e))
    except Exception:
        return dict(vti=np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. AUX (277) — paper-style FULL feature set
#     mean + std + var dla każdego parametru (1-based indeksy)
#
# Paper: "277 features from 14 audio parameters, including mean, std, var".
#   MFCC×20 + MFCC∆×20 + MFCC∆∆×20  → 60+60+60 = 180
#   Chroma CENS×12                  → 36
#   Tonnetz×6  (paper podaje 7; librosa zwraca 6 — dokumentowana rozbieżność) → 18
#   Spectral Contrast×8 (librosa n_bands=7 ⇒ 8 wierszy) → 24
#   Scalary×6: rms, spec_centroid, spec_bw, spec_rolloff, spec_flatness, zcr → 18
#   Tempo (BPM, brak mean/std/var)  → 1
#   ─────────────────────────────────
#   Suma: 180+36+18+24+18+1 = 277
# ═══════════════════════════════════════════════════════════════════════════════

N_MFCC_EXT       = 20
N_CONTRAST_BANDS = 7   # → 8 wierszy w wyjściu spectral_contrast


def _msv(vec, name):
    """Emit mean/std/var dla wektora w paper-style namingu (`{stat}_{name}`)."""
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return {f"mean_{name}": np.nan,
                f"std_{name}":  np.nan,
                f"var_{name}":  np.nan}
    return {f"mean_{name}": float(np.mean(arr)),
            f"std_{name}":  float(np.std(arr)),
            f"var_{name}":  float(np.var(arr))}


def _msv_nan(name):
    return {f"mean_{name}": np.nan,
            f"std_{name}":  np.nan,
            f"var_{name}":  np.nan}


def _aux_features(y, S):
    """277 cech paper-style: mean/std/var dla każdego parametru.

    Oryginalna lista 34 cech użytkownika to podzbiór tego zestawu —
    pełna iteracja `range(1, N+1)` × `{mean,std,var}` pokrywa wszystkie nazwy.
    """
    out = {}
    y_f32 = y.astype(np.float32)

    # ── MFCC, MFCC∆, MFCC∆∆  (20 × 3 stats × 3 series = 180 cech)
    try:
        mfcc20 = librosa.feature.mfcc(y=y_f32, sr=SR, n_mfcc=N_MFCC_EXT)
        d1     = librosa.feature.delta(mfcc20, order=1)
        d2     = librosa.feature.delta(mfcc20, order=2)
        for n in range(1, N_MFCC_EXT + 1):
            out.update(_msv(mfcc20[n - 1], f"mfcc{n}"))
            out.update(_msv(d1[n - 1],     f"mfcc_delta_{n}"))
            out.update(_msv(d2[n - 1],     f"mfcc_delta2_{n}"))
    except Exception:
        for n in range(1, N_MFCC_EXT + 1):
            out.update(_msv_nan(f"mfcc{n}"))
            out.update(_msv_nan(f"mfcc_delta_{n}"))
            out.update(_msv_nan(f"mfcc_delta2_{n}"))

    # ── Chroma CENS (12 binów × 3 = 36 cech)
    try:
        cens = librosa.feature.chroma_cens(y=y_f32, sr=SR, hop_length=HOP)
        for n in range(1, 13):
            out.update(_msv(cens[n - 1], f"chroma_cens_{n}"))
    except Exception:
        for n in range(1, 13):
            out.update(_msv_nan(f"chroma_cens_{n}"))

    # ── Tonnetz (6 dim × 3 = 18 cech)
    try:
        y_harm = librosa.effects.harmonic(y_f32)
        tonn   = librosa.feature.tonnetz(y=y_harm, sr=SR)
        n_dims = tonn.shape[0]
        for n in range(1, n_dims + 1):
            out.update(_msv(tonn[n - 1], f"tonnetz_{n}"))
        for n in range(n_dims + 1, 7):  # uzupełnij do 6 jeśli mniej
            out.update(_msv_nan(f"tonnetz_{n}"))
    except Exception:
        for n in range(1, 7):
            out.update(_msv_nan(f"tonnetz_{n}"))

    # ── Spectral contrast (8 pasm × 3 = 24 cechy)
    try:
        sc = librosa.feature.spectral_contrast(S=S, sr=SR,
                                                n_bands=N_CONTRAST_BANDS)
        n_bands = sc.shape[0]  # = N_CONTRAST_BANDS + 1 = 8
        for n in range(1, n_bands + 1):
            out.update(_msv(sc[n - 1], f"spectral_contrast_{n}"))
        for n in range(n_bands + 1, N_CONTRAST_BANDS + 2):
            out.update(_msv_nan(f"spectral_contrast_{n}"))
    except Exception:
        for n in range(1, N_CONTRAST_BANDS + 2):
            out.update(_msv_nan(f"spectral_contrast_{n}"))

    # ── Scalary (6 parametrów × 3 = 18 cech)
    try:
        out.update(_msv(librosa.feature.rms(y=y, hop_length=HOP)[0], "rms"))
    except Exception:
        out.update(_msv_nan("rms"))
    try:
        out.update(_msv(librosa.feature.spectral_centroid(S=S, sr=SR)[0],
                        "spectral_centroid"))
    except Exception:
        out.update(_msv_nan("spectral_centroid"))
    try:
        out.update(_msv(librosa.feature.spectral_bandwidth(S=S, sr=SR)[0],
                        "spectral_bandwidth"))
    except Exception:
        out.update(_msv_nan("spectral_bandwidth"))
    try:
        out.update(_msv(librosa.feature.spectral_rolloff(S=S, sr=SR)[0],
                        "spectral_rolloff"))
    except Exception:
        out.update(_msv_nan("spectral_rolloff"))
    try:
        out.update(_msv(librosa.feature.spectral_flatness(y=y_f32,
                                                          n_fft=N_FFT,
                                                          hop_length=HOP)[0],
                        "spectral_flatness"))
    except Exception:
        out.update(_msv_nan("spectral_flatness"))
    try:
        out.update(_msv(librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0],
                        "zcr"))
    except Exception:
        out.update(_msv_nan("zcr"))

    # ── Tempo (1 scalar — brak mean/std/var)
    try:
        tempo_val, _ = librosa.beat.beat_track(y=y_f32, sr=SR)
        out["tempo"] = float(np.atleast_1d(tempo_val).flatten()[0])
    except Exception:
        out["tempo"] = np.nan

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HELPER — ekstrakcja z gotowego segmentu sygnału
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_window(y_win: np.ndarray) -> dict:
    """Oblicza 464 cechy dla segmentu sygnału (bez I/O)."""
    snd = parselmouth.Sound(y_win.astype(np.float64), sampling_frequency=SR)
    S   = np.abs(librosa.stft(y_win, n_fft=N_FFT, hop_length=HOP))

    out = {}
    # 1.–4. baseline 150 cech (extract_133)
    out.update(_jitter(snd))
    out.update(_shimmer(snd))
    out.update(_hnr(snd))
    out.update(_formants(snd))
    out.update(_mfcc(y_win))
    direct, rms = _direct(y_win, S)
    out.update(direct)
    out.update(_syllables(y_win))
    out.update(_repetition(rms))
    out.update(_spec_entropy_variability(S))
    out.update(_f0(snd))
    out.update(_formant_bandwidths(snd))
    out.update(_cpp(y_win))
    out.update(_spectral_tilt(S))

    # 6. EXT Tier-1 (17)
    out.update(_band_energy_ratios(S))
    out.update(_hammarberg(S))
    out.update(_loudness_db(y_win))
    out.update(_glottal_source(y_win, snd))
    out.update(_spectral_flatness(y_win))

    # 7. EXT Tier-2 (11)
    out.update(_cpps(y_win))
    out.update(_gne(y_win))
    out.update(_f0_contour(snd))
    out.update(_pause_rate(y_win, snd))

    # 8. EXT Tier-3 (9)
    out.update(_dfa(snd))
    out.update(_ppe(snd))
    out.update(_shr(y_win, snd))
    out.update(_lpc_residual(y_win))
    out.update(_spi(S))
    out.update(_vti(S))

    # 8b. AUX (277) — paper-style mean/std/var
    out.update(_aux_features(y_win, S))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GŁÓWNA FUNKCJA — całe nagranie
# ═══════════════════════════════════════════════════════════════════════════════

def extract_n(path: str) -> dict:
    """
    Wczytuje plik audio i oblicza 464 cechy głosu (187 clinical-grade + 277 AUX paper-style).

    Parametry
    ---------
    path : str
        Ścieżka do pliku audio (WAV, M4A lub inny format obsługiwany przez librosa).

    Zwraca
    ------
    dict
        Płaski słownik {nazwa_cechy: float} z 464 kluczami.
        Przy błędzie pojedynczej grupy zwraca NaN dla dotkniętej grupy
        (pozostałe są obliczone normalnie).
    """
    y, _, _ = _load(path)
    return _extract_window(y)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def extract_n_sliding(
    path: str,
    window_sec: float = 3.0,
    hop_sec: float = 1.0,
    min_window_sec: float = 1.0,
) -> list[dict]:
    """
    Wyciąga 464 cechy w oknie przesuwnym i zwraca listę słowników.

    Parametry
    ---------
    path : str
        Ścieżka do pliku audio.
    window_sec : float
        Długość okna w sekundach (domyślnie 3.0 s).
    hop_sec : float
        Krok przesuwania okna w sekundach (domyślnie 1.0 s).
    min_window_sec : float
        Minimalna długość ostatniego (przyciętego) okna — krótsze są pomijane.

    Zwraca
    ------
    list[dict]
        Lista słowników cech; każdy dict zawiera dodatkowo:
          - ``window_start`` : float  — czas początku okna [s]
          - ``window_end``   : float  — czas końca okna [s]
          - ``window_idx``   : int    — indeks okna (od 0)
    """
    y, _, _ = _load(path)

    win_samples = int(window_sec * SR)
    hop_samples = int(hop_sec    * SR)
    min_samples = int(min_window_sec * SR)
    total       = len(y)

    if total < min_samples:
        raise ValueError(
            f"Plik '{path}' jest krótszy niż min_window_sec={min_window_sec}s "
            f"({total/SR:.2f}s dostępne)."
        )

    results = []
    idx     = 0
    start   = 0

    while start < total:
        end    = min(start + win_samples, total)
        length = end - start

        if length < min_samples:
            break

        y_win  = y[start:end]
        feats  = _extract_window(y_win)

        feats["window_start"] = round(start / SR, 4)
        feats["window_end"]   = round(end   / SR, 4)
        feats["window_idx"]   = idx

        results.append(feats)

        start += hop_samples
        idx   += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 11b. EKSTRAKCJA Z PREFIKSU PLIKU — librosa.load(duration=...)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_n_sec(path: str, sec: float = 1.0) -> dict:
    """
    Wczytuje TYLKO pierwsze `sec` sekund pliku i oblicza 464 cechy.

    Wewnętrznie: `librosa.load(path, sr=SR, mono=True, duration=sec)` — nie czyta
    całego pliku, więc dla długich nagrań szybsze niż `extract_n(path)`.

    Parametry
    ---------
    path : str
        Ścieżka do pliku audio.
    sec : float
        Długość prefiksu w sekundach (domyślnie 1.0 s).

    Zwraca
    ------
    dict
        Płaski słownik {nazwa_cechy: float} z 464 kluczami (jak `extract_n`).

    Uwagi
    -----
    Krótkie wartości `sec` (< 2 s) powodują wiele NaN-ów w cechach Tier-3
    (DFA, SHR, beat_track/tempo) — wymagają wystarczająco długiego sygnału.
    """
    y, _ = librosa.load(path, sr=SR, mono=True, duration=sec)
    if len(y) == 0:
        raise ValueError(f"Załadowano 0 próbek z '{path}' (sec={sec}).")
    y = y / (np.max(np.abs(y)) + 1e-9)
    return _extract_window(y)


# Aliasy zgodne z extract_133 — żeby nie psuć kodu, który już importuje stare nazwy
extract_133         = extract_n
extract_133_sliding = extract_n_sliding


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CLI — szybki test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Użycie: python extract_n.py <plik.wav|.m4a>")
        sys.exit(1)

    feats = extract_n(path)
    print(f"Liczba cech: {len(feats)}")

    groups = {
        "Jitter":       [k for k in feats if "jitter"      in k],
        "Shimmer":      [k for k in feats if "shimmer"     in k],
        "HNR":          [k for k in feats if k.startswith("hnr_")],
        "Formanty":     [k for k in feats if k.startswith(("F1","F2","F3"))],
        "MFCC":         [k for k in feats if "mfcc"        in k],
        "Direct":       [k for k in feats if any(t+"_" in k for t in
                          ["rms","centroid","bandwidth","rolloff","zcr"])
                          or k in ("env_mean","env_std","mod_std","mod_entropy","aci")],
        "Sylaby":       [k for k in feats if "syllable"    in k],
        "Repetycja":    [k for k in feats if "repetition"  in k],
        "Spec entr":    [k for k in feats if "spec_entropy" in k],
        "F0":           [k for k in feats if k.startswith("f0_") or k in ("voicing_ratio","voice_breaks")],
        "Form. BW":     [k for k in feats if k.startswith(("B1_","B2_","B3_"))],
        "CPP":          [k for k in feats if k.startswith("cpp_")],
        "Tilt":         [k for k in feats if k in ("spectral_slope","alpha_ratio")],
        # ── EXT
        "BER":          [k for k in feats if k.startswith("ber_")],
        "Hammarberg":   [k for k in feats if "hammarberg"  in k],
        "Loudness dB":  [k for k in feats if k.startswith("loudness_")],
        "H1H2/H1A":     [k for k in feats if k.startswith(("h1h2","h1a1","h1a3"))],
        "Flatness":     [k for k in feats if k.startswith("flatness_")],
        "CPPS":         [k for k in feats if k.startswith("cpps_")],
        "GNE":          [k for k in feats if k.startswith("gne")],
        "F0 contour":   [k for k in feats if k.startswith("f0_slope") or k == "f0_traj_entropy"],
        "Pause/rate":   [k for k in feats if k in ("pause_count","pause_mean_dur",
                                                    "articulation_rate","voiced_ratio")],
        "DFA":          [k for k in feats if k.startswith("dfa_")],
        "PPE":          [k for k in feats if k == "ppe"],
        "SHR":          [k for k in feats if k.startswith("shr_")],
        "LPC residual": [k for k in feats if k.startswith("lpc_res_")],
        "SPI":          [k for k in feats if k == "spi"],
        "VTI":          [k for k in feats if k == "vti"],
        # ── AUX (277) — paper-style mean/std/var
        "AUX MFCC":         [k for k in feats
                              if any(k.startswith(p + "mfcc")
                                     for p in ("mean_","std_","var_"))
                              and "_delta" not in k],
        "AUX MFCC Δ":       [k for k in feats if "mfcc_delta_" in k
                              and "delta2" not in k],
        "AUX MFCC ΔΔ":      [k for k in feats if "mfcc_delta2_" in k],
        "AUX Chroma CENS":  [k for k in feats if "chroma_cens_" in k],
        "AUX Tonnetz":      [k for k in feats
                              if any(k.startswith(p + "tonnetz_")
                                     for p in ("mean_","std_","var_"))],
        "AUX Spec.contr":   [k for k in feats
                              if any(k.startswith(p + "spectral_contrast_")
                                     for p in ("mean_","std_","var_"))],
        "AUX Scalary":      [k for k in feats
                              if any(k == f"{s}_{name}"
                                     for s in ("mean","std","var")
                                     for name in ("rms","zcr",
                                                  "spectral_centroid",
                                                  "spectral_bandwidth",
                                                  "spectral_rolloff",
                                                  "spectral_flatness"))],
        "AUX Tempo":        [k for k in feats if k == "tempo"],
    }
    for grp, keys in groups.items():
        print(f"\n── {grp} ({len(keys)}) {'─' * max(0, 38 - len(grp))}")
        for k in keys:
            v = feats[k]
            try:
                ok = np.isfinite(v)
            except Exception:
                ok = False
            print(f"  {k:<35s} {v:>12.5f}" if ok else f"  {k:<35s} {'NaN':>12}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SYGNAŁ Z PAMIĘCI → 261 CECH (zestaw po odsianiu duplikatów)
# ═══════════════════════════════════════════════════════════════════════════════

_FEATS_261_CACHE = None
_FEATS_261_CSV = ("/experiments/dgutowska/wavlm-ecapa/analytics_paper_2.0/"
                  "corr_abs_pearson_full_population.csv")


def feats_261() -> list[str]:
    """Nazwy 261 cech, posortowane alfabetycznie — kolejność jak przy treningu modeli.

    Zestaw pochodzi z `analytics_paper_2.0/2-features-large-population.ipynb`:
    z 464 cech odrzucono 24 martwe (same NaN albo zerowa wariancja) i 179
    near-duplikatów (Spearman > 0.99 wobec cechy wcześniejszej na liście).
    """
    global _FEATS_261_CACHE
    if _FEATS_261_CACHE is None:
        import pandas as pd
        _FEATS_261_CACHE = sorted(set(pd.read_csv(_FEATS_261_CSV)["feature"]))
    return list(_FEATS_261_CACHE)


def extract_n_signal(signal, sr: int = SR, normalize: bool = True,
                     feats: list[str] | None = None, as_array: bool = False):
    """261 cech z sygnału trzymanego w pamięci, bez I/O.

    Parametry
    ---------
    signal : np.ndarray
        Przebieg mono. Przy `sr != SR` zostanie przepróbkowany do 16 kHz.
    sr : int
        Częstotliwość próbkowania `signal`.
    normalize : bool
        Normalizacja szczytowa `y / max|y|`, tak jak robi `_load` przy wczytywaniu
        z pliku. Zostaw `True`, jeśli wynik ma się zgadzać z `extract_n(path)`.
    feats : list[str] | None
        Lista cech do zwrócenia. `None` → `feats_261()`.
    as_array : bool
        `True` → wektor `np.ndarray` w kolejności `feats`, gotowy dla modelu.
        `False` → dict {nazwa: wartość}.

    Zwraca
    ------
    dict | np.ndarray
        261 cech. Przy błędzie Praata dotknięta grupa ma NaN, reszta jest policzona.

    Uwaga
    -----
    Wewnętrznie liczone są wszystkie 464 cechy i dopiero potem wybierane 261 —
    ekstraktor działa grupami, więc policzenie podzbioru nie jest tańsze.
    """
    y = np.asarray(signal, dtype=np.float64).ravel()
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if normalize:
        y = y / (np.max(np.abs(y)) + 1e-9)

    out = _extract_window(y.astype(np.float32))
    names = feats_261() if feats is None else list(feats)

    if as_array:
        return np.array([out.get(k, np.nan) for k in names], dtype=float)
    return {k: out.get(k, np.nan) for k in names}


_FEATS_261_RESULTS_CSV = "/experiments/dgutowska/wavlm-ecapa/results/feats_261.csv"
_FEATS_261_RESULTS_CACHE = None


def feats_261_results() -> list[str]:
    """Nazwy 261 cech wczytane z `results/feats_261.csv` (kolumna `feature`)."""
    global _FEATS_261_RESULTS_CACHE
    if _FEATS_261_RESULTS_CACHE is None:
        import pandas as pd
        _FEATS_261_RESULTS_CACHE = sorted(set(pd.read_csv(_FEATS_261_RESULTS_CSV)["feature"]))
    return list(_FEATS_261_RESULTS_CACHE)


def extract_n_signal_261(signal, sr: int = SR, normalize: bool = True, as_array: bool = False):
    """261 cech z sygnału trzymanego w pamięci — lista cech z `results/feats_261.csv`.

    Wrapper wokół `extract_n_signal`, z `feats` ustawionym na `feats_261_results()`.
    """
    return extract_n_signal(signal, sr=sr, normalize=normalize,
                             feats=feats_261_results(), as_array=as_array)
