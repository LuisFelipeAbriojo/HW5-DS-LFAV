"""Decodificación de audio a float32 mono con PyAV.

No se usa faster_whisper.audio.decode_audio: en faster-whisper 1.2.1 llama a av.open(..., metadata_errors=...),
argumento que PyAV >= 15 eliminó (TypeError). Decodificamos aquí y pasamos el arreglo al modelo.
"""
from __future__ import annotations

import av
import numpy as np


def decode(path: str, sampling_rate: int = 16000) -> np.ndarray:
    chunks = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            for r in resampler.resample(frame):
                chunks.append(r.to_ndarray().reshape(-1))
        for r in resampler.resample(None):
            chunks.append(r.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
