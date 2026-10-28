"""Escalera de degradación reproducible.

L0  tal cual se grabó (decodificado a 16 kHz mono)
L1  nota de voz de WhatsApp: Opus a 16 kbit/s, mono (PyAV/libopus; respaldo MP3 si no hay libopus)
L2  llamada telefónica: pasa-banda 300-3400 Hz (Butterworth) y remuestreo a 8 kHz
L3  ambiente ruidoso: ruido ambiente mezclado a una SNR objetivo y luego el códec de L1
    (el micrófono capta el ruido ANTES de comprimir: por eso el ruido va antes del códec)

La SNR se MIDE, no se supone: potencia de la voz solo en tramas activas (VAD por energía) contra la
potencia del ruido escalado en esas mismas tramas.
"""
from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from .audio import decode as decode_audio
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000


def load(path: str, sr: int = SR) -> np.ndarray:
    return decode_audio(str(path), sampling_rate=sr).astype(np.float32)


def codec_roundtrip(x: np.ndarray, sr: int, codec: str, bitrate: int, container: str) -> np.ndarray:
    """Codifica y decodifica en memoria: el artefacto del códec queda en la señal."""
    buf = io.BytesIO()
    with av.open(buf, "w", format=container) as out:
        rate = 48000 if codec == "libopus" else sr
        st = out.add_stream(codec, rate=rate)
        st.bit_rate = bitrate
        st.layout = "mono"
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        if rate != sr:
            pcm = (resample_poly(pcm.astype(np.float32), rate, sr)).clip(-32768, 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
        frame.sample_rate = rate
        for pkt in st.encode(frame):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    buf.seek(0)
    chunks = []
    with av.open(buf, "r") as inp:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frm in inp.decode(audio=0):
            for r in resampler.resample(frm):
                chunks.append(r.to_ndarray().reshape(-1))
    y = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return y[: len(x)].astype(np.float32)


def whatsapp(x: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    c = cfg["ladder"]["L1"]
    try:
        return codec_roundtrip(x, SR, c["codec"], c["bitrate"], c["container"]), {"codec": c["codec"], "bitrate": c["bitrate"]}
    except Exception as exc:  # PyAV sin libopus
        y = codec_roundtrip(x, SR, c["fallback_codec"], c["fallback_bitrate"], "mp3")
        return y, {"codec": c["fallback_codec"], "bitrate": c["fallback_bitrate"], "fallback_reason": str(exc)}


def telephone(x: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, dict]:
    c = cfg["ladder"]["L2"]
    sos = butter(c["filter_order"], [c["low_hz"], c["high_hz"]], btype="bandpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, x)
    y = resample_poly(y, c["resample_hz"], SR).astype(np.float32)
    return y, c["resample_hz"], {"band_hz": [c["low_hz"], c["high_hz"]], "sr": c["resample_hz"]}


def active_frames(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Máscara por muestra de tramas con voz (energía sobre el percentil configurado)."""
    n = int(SR * cfg["ladder"]["vad_frame_ms"] / 1000)
    frames = len(x) // n
    if frames == 0:
        return np.ones_like(x, dtype=bool)
    e = (x[: frames * n].reshape(frames, n) ** 2).mean(axis=1)
    thr = np.percentile(e, cfg["ladder"]["vad_energy_percentile"])
    mask = np.repeat(e > thr, n)
    return np.concatenate([mask, np.zeros(len(x) - len(mask), dtype=bool)])


def noise_like(n: int, cfg: dict, noise_path: Path | None) -> tuple[np.ndarray, str]:
    if noise_path and Path(noise_path).exists():
        nz = load(str(noise_path))
        reps = int(np.ceil(n / len(nz)))
        return np.tile(nz, reps)[:n], f"grabado:{Path(noise_path).name}"
    # respaldo reproducible: ruido rosa (1/f) con semilla. Declararlo en el informe.
    rng = np.random.default_rng(cfg["ladder"]["L3"]["noise_seed"])
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = np.fft.rfft(white) / np.sqrt(np.maximum(f, 20.0))
    pink = np.fft.irfft(spec, n)
    return (pink / np.abs(pink).max()).astype(np.float32), "ruido_rosa_sintetico"


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float, cfg: dict) -> tuple[np.ndarray, dict]:
    mask = active_frames(x, cfg)
    p_speech = float(np.mean(x[mask] ** 2))
    p_noise = float(np.mean(noise[mask] ** 2)) or 1e-12
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    scaled = noise * gain
    measured = 10 * np.log10(p_speech / float(np.mean(scaled[mask] ** 2)))
    y = x + scaled
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y / peak * 0.99
    return y.astype(np.float32), {"snr_target_db": snr_db, "snr_measured_db": round(float(measured), 2),
                                  "speech_frames_pct": round(float(100 * mask.mean()), 1)}


def build_level(x: np.ndarray, level: str, cfg: dict, noise_path: Path | None = None) -> tuple[np.ndarray, int, dict]:
    if level == "L0":
        return x, SR, {}
    if level == "L1":
        y, meta = whatsapp(x, cfg)
        return y, SR, meta
    if level == "L2":
        return telephone(x, cfg)
    spec = cfg["ladder"].get(level) or cfg["ladder"]["hardening"][level]
    noise, src = noise_like(len(x), cfg, noise_path)
    noisy, meta = mix_at_snr(x, noise, spec["snr_db"], cfg)
    meta["noise_source"] = src
    y, sr, m2 = build_level(noisy, spec["base"], cfg)
    meta.update({f"base_{k}": v for k, v in m2.items()})
    meta["base"] = spec["base"]
    return y, sr, meta


def write(path: Path, y: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
