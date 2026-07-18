"""
Lightweight MFCC-based voice similarity module.
Pure numpy/scipy, no torch, no librosa required.
Designed for speaker verification against a stored baseline.
"""
import math
import wave
import struct
import os
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.fft import dct
from scipy.signal import medfilt2d


# ─── Mel Filterbank ───────────────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_filters: int = 26, n_fft: int = 512, sr: int = 16000) -> np.ndarray:
    """Create a mel-scale triangular filterbank matrix (n_filters x (n_fft//2+1))."""
    low_mel = _hz_to_mel(80)
    high_mel = _hz_to_mel(min(sr / 2, 8000))
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i, j] = (right - j) / (right - center)
    return fb


# ─── MFCC Extraction ─────────────────────────────────────────────────────────

def extract_mfcc(
    samples: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 26,
    pre_emphasis: float = 0.97,
) -> np.ndarray:
    """
    Extract MFCC features from raw audio samples.
    Returns shape (n_frames, n_mfcc).
    """
    # Pre-emphasis
    emphasized = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

    # Framing
    frame_length = n_fft
    n_frames = 1 + (len(emphasized) - frame_length) // hop_length
    if n_frames <= 0:
        return np.zeros((1, n_mfcc))

    indices = np.arange(frame_length)[None, :] + np.arange(n_frames)[:, None] * hop_length
    frames = emphasized[indices]

    # Hamming window
    window = np.hamming(frame_length)
    frames = frames * window

    # FFT
    mag = np.abs(np.fft.rfft(frames, n=n_fft))
    power = (mag ** 2) / n_fft

    # Mel filterbank
    fb = _mel_filterbank(n_mels, n_fft, sr)
    mel_spec = np.dot(power, fb.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # DCT to get MFCCs
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc


# ─── HPSS (Harmonic-Percussive Source Separation) ────────────────────────────

def hpss_vocal_enhance(
    samples: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    kernel_size: int = 31,
) -> np.ndarray:
    """
    Simple HPSS to suppress percussive/noise components and keep harmonic (vocal).
    Returns enhanced audio samples (same length as input).
    """
    # STFT
    n_frames = 1 + (len(samples) - n_fft) // hop_length
    if n_frames <= 0:
        return samples

    indices = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop_length
    frames = samples[indices] * np.hanning(n_fft)
    stft = np.fft.rfft(frames, n=n_fft)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Median filtering: horizontal (time) for harmonic, vertical (freq) for percussive
    k = min(kernel_size, mag.shape[0]) | 1  # ensure odd
    harmonic_mag = medfilt2d(mag, kernel_size=(k, 1))
    percussive_mag = medfilt2d(mag, kernel_size=(1, k))

    # Soft mask for harmonic
    total = harmonic_mag + percussive_mag + 1e-10
    harmonic_mask = harmonic_mag / total

    # Apply mask
    enhanced_stft = stft * harmonic_mask

    # iSTFT (overlap-add)
    output = np.zeros(len(samples))
    window_sum = np.zeros(len(samples))
    win = np.hanning(n_fft)
    for i in range(n_frames):
        frame = np.fft.irfft(enhanced_stft[i], n=n_fft)
        start = i * hop_length
        output[start:start + n_fft] += frame * win
        window_sum[start:start + n_fft] += win ** 2

    # Normalize
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum
    return output


# ─── VAD Segmentation ─────────────────────────────────────────────────────────

def vad_segments(
    samples: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 30,
    energy_threshold_db: float = -35.0,
    min_speech_ms: int = 300,
    min_silence_ms: int = 200,
) -> List[Tuple[int, int]]:
    """
    Simple energy-based VAD. Returns list of (start_sample, end_sample) speech segments.
    """
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_len
    if n_frames <= 0:
        return [(0, len(samples))]

    # Frame energies in dB
    energies = []
    for i in range(n_frames):
        frame = samples[i * frame_len:(i + 1) * frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        energies.append(20.0 * np.log10(rms + 1e-10))

    energies = np.array(energies)

    # Adaptive threshold: use provided or adapt to content
    threshold = max(energy_threshold_db, float(np.percentile(energies, 25)) + 6.0)

    # Flag speech frames
    speech_flags = energies >= threshold

    # Merge short silences, remove short speech
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    # Fill short silence gaps
    i = 0
    while i < len(speech_flags):
        if not speech_flags[i]:
            gap_start = i
            while i < len(speech_flags) and not speech_flags[i]:
                i += 1
            if i - gap_start < min_silence_frames:
                speech_flags[gap_start:i] = True
        else:
            i += 1

    # Extract segments
    segments = []
    in_speech = False
    start = 0
    for i, flag in enumerate(speech_flags):
        if flag and not in_speech:
            start = i
            in_speech = True
        elif not flag and in_speech:
            if i - start >= min_speech_frames:
                segments.append((start * frame_len, i * frame_len))
            in_speech = False
    if in_speech and len(speech_flags) - start >= min_speech_frames:
        segments.append((start * frame_len, len(samples)))

    if not segments:
        return [(0, len(samples))]
    return segments


# ─── Similarity ───────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mfcc_mean_vec(mfcc: np.ndarray) -> np.ndarray:
    """Compute extended speaker embedding: mean + std of MFCC (2*n_mfcc dims)."""
    if mfcc.ndim == 1:
        return np.concatenate([mfcc, np.zeros_like(mfcc)])
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)
    return np.concatenate([mean, std])


def mfcc_similarity(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
    """Compute similarity between two MFCC matrices using mean vectors."""
    vec_a = mfcc_mean_vec(mfcc_a)
    vec_b = mfcc_mean_vec(mfcc_b)
    return cosine_similarity(vec_a, vec_b)


# ─── High-level Pipeline ─────────────────────────────────────────────────────

def load_wav_samples(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load mono 16-bit WAV as float32 numpy array and sample rate."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if width == 2:
        fmt = f"<{n_frames * n_channels}h"
        data = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
    elif width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise RuntimeError(f"Unsupported sample width: {width}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data, sr


def extract_speaker_embedding(
    wav_path: str,
    use_hpss: bool = False,
    n_mfcc: int = 13,
) -> Optional[np.ndarray]:
    """
    Extract a simple speaker embedding (mean MFCC vector) from a WAV file.
    Optionally applies HPSS to suppress background music first.
    Returns None if extraction fails.
    """
    try:
        samples, sr = load_wav_samples(wav_path)
        if len(samples) < sr * 0.3:  # too short
            return None

        if use_hpss:
            samples = hpss_vocal_enhance(samples, sr=sr)

        # VAD: only use speech segments
        segments = vad_segments(samples, sr=sr)
        speech_samples = np.concatenate([samples[s:e] for s, e in segments])

        if len(speech_samples) < sr * 0.2:
            return None

        mfcc = extract_mfcc(speech_samples, sr=sr, n_mfcc=n_mfcc)
        return mfcc_mean_vec(mfcc)
    except Exception:
        return None


def compare_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> Dict:
    """
    Compare two speaker embeddings.
    Returns dict with similarity score (0-100) and label.
    """
    sim = cosine_similarity(emb_a, emb_b)
    # Map cosine similarity to 0-100 score
    # For mean-MFCC embeddings on real speech:
    #   same speaker, same session: 0.95-0.99
    #   same speaker, different session: 0.88-0.96
    #   different speaker: 0.70-0.90
    # Use a tighter sigmoid-like mapping centered around 0.90
    if sim >= 0.98:
        score = 95.0 + (sim - 0.98) / 0.02 * 5.0
    elif sim >= 0.92:
        score = 70.0 + (sim - 0.92) / 0.06 * 25.0
    elif sim >= 0.85:
        score = 45.0 + (sim - 0.85) / 0.07 * 25.0
    elif sim >= 0.75:
        score = 20.0 + (sim - 0.75) / 0.10 * 25.0
    else:
        score = max(0.0, sim / 0.75 * 20.0)

    score = max(0.0, min(100.0, score))

    if score >= 75:
        label = "像你"
    elif score >= 50:
        label = "有点像你"
    elif score >= 30:
        label = "不太像你"
    else:
        label = "底纹不确定"

    return {
        "cosine_sim": round(sim, 4),
        "score": round(score, 1),
        "label": label,
    }


def segment_analysis(
    wav_path: str,
    baseline_embedding: Optional[np.ndarray] = None,
    use_hpss: bool = False,
    n_mfcc: int = 13,
) -> List[Dict]:
    """
    Analyze audio by segments. For each speech segment, extract embedding
    and optionally compare to baseline.
    Returns list of segment results.
    """
    try:
        samples, sr = load_wav_samples(wav_path)
    except Exception:
        return []

    if use_hpss:
        try:
            samples = hpss_vocal_enhance(samples, sr=sr)
        except Exception:
            pass

    segments = vad_segments(samples, sr=sr)
    results = []

    for i, (start, end) in enumerate(segments):
        seg_samples = samples[start:end]
        duration_ms = int((end - start) / sr * 1000)

        if duration_ms < 300:
            results.append({
                "segment": i,
                "start_ms": int(start / sr * 1000),
                "end_ms": int(end / sr * 1000),
                "duration_ms": duration_ms,
                "comparable": False,
                "reason": "太短",
            })
            continue

        mfcc = extract_mfcc(seg_samples, sr=sr, n_mfcc=n_mfcc)
        emb = mfcc_mean_vec(mfcc)

        result = {
            "segment": i,
            "start_ms": int(start / sr * 1000),
            "end_ms": int(end / sr * 1000),
            "duration_ms": duration_ms,
            "comparable": True,
            "embedding": emb,
        }

        if baseline_embedding is not None:
            comp = compare_embeddings(emb, baseline_embedding)
            result.update(comp)

        results.append(result)

    return results
