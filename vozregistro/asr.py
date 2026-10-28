"""Transcripción con faster-whisper, con evidencia medida del dispositivo usado.

El dispositivo NO se toma de la configuración: se lee del objeto CTranslate2 ya cargado
(`model.model.device`) y se respalda con la VRAM medida por NVML durante la transcripción.
"""
from __future__ import annotations

import inspect
import threading
import time
from functools import lru_cache

import ctranslate2
import faster_whisper
from faster_whisper import WhisperModel
from .audio import decode as decode_audio

try:
    import pynvml  # paquete nvidia-ml-py
except ImportError:  # pragma: no cover
    pynvml = None


def cuda_device_count() -> int:
    try:
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if cuda_device_count() > 0 else "cpu"
    return requested


def resolve_compute_type(device: str, requested: str) -> str:
    supported = ctranslate2.get_supported_compute_types(device)
    if requested in supported:
        return requested
    return "int8" if "int8" in supported else "float32"


def glossary_support() -> dict:
    """Qué mecanismos de glosario acepta la versión INSTALADA (no la documentación)."""
    params = inspect.signature(WhisperModel.transcribe).parameters
    return {"faster_whisper": faster_whisper.__version__,
            "initial_prompt": "initial_prompt" in params,
            "hotwords": "hotwords" in params}


def release_models():
    """Libera los modelos cacheados (el benchmark lo usa para medir cada configuración desde cero)."""
    import gc
    get_model.cache_clear()
    gc.collect()


@lru_cache(maxsize=4)
def get_model(size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(size, device=device, compute_type=compute_type)


class VramMonitor:
    """Muestrea la memoria de la GPU en un hilo; reporta pico del proceso y pico del dispositivo sobre la base."""

    def __init__(self, interval: float = 0.05, index: int = 0):
        self.interval = interval
        self.index = index
        self.available = False
        self.baseline_mb = None
        self.peak_device_mb = None
        self.peak_process_mb = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self, handle, pid):
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
        proc = None
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if p.pid == pid and p.usedGpuMemory:
                    proc = p.usedGpuMemory / 2**20
        except pynvml.NVMLError:
            pass
        return used, proc

    def __enter__(self):
        if pynvml is None:
            return self
        try:
            import os
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            pid = os.getpid()
            self.available = True
            self.baseline_mb, _ = self._sample(handle, pid)
            self.peak_device_mb = self.baseline_mb

            def loop():
                while not self._stop.is_set():
                    used, proc = self._sample(handle, pid)
                    self.peak_device_mb = max(self.peak_device_mb, used)
                    if proc is not None:
                        self.peak_process_mb = max(self.peak_process_mb or 0, proc)
                    time.sleep(self.interval)

            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
        except Exception:
            self.available = False
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return False

    def report(self) -> dict:
        if not self.available:
            return {"nvml": False, "peak_vram_mb": None, "delta_vram_mb": None, "process_vram_mb": None}
        return {"nvml": True,
                "peak_vram_mb": round(self.peak_device_mb, 1),
                "delta_vram_mb": round(self.peak_device_mb - self.baseline_mb, 1),
                "process_vram_mb": round(self.peak_process_mb, 1) if self.peak_process_mb else None}


def transcribe(audio_path: str, asr_cfg: dict, *, model_size: str | None = None, device: str | None = None,
               compute_type: str | None = None, glossary_terms: list[str] | None = None,
               glossary_mode: str | None = None, vram_interval: float = 0.05) -> dict:
    size = model_size or asr_cfg["model"]
    dev = resolve_device(device or asr_cfg["device"])
    ctype = resolve_compute_type(dev, compute_type or asr_cfg["compute_type"])
    mode = (glossary_mode or asr_cfg.get("glossary_mode", "none")) if glossary_terms else "none"
    support = glossary_support()

    kwargs = dict(language=asr_cfg["language"], beam_size=asr_cfg["beam_size"],
                  vad_filter=asr_cfg["vad_filter"], word_timestamps=asr_cfg["word_timestamps"])
    if mode == "hotwords" and not support["hotwords"]:
        mode = "initial_prompt"  # la versión instalada no lo soporta: se registra el mecanismo real
    if mode == "hotwords":
        kwargs["hotwords"] = ", ".join(glossary_terms)
    elif mode == "initial_prompt":
        kwargs["initial_prompt"] = "Vocabulario: " + ", ".join(glossary_terms) + "."

    audio = decode_audio(audio_path, sampling_rate=16000)
    audio_s = len(audio) / 16000

    # El monitor abarca carga + inferencia: delta_vram_mb incluye los pesos si el modelo no estaba en caché.
    with VramMonitor(interval=vram_interval) as mon:
        t_load = time.perf_counter()
        model = get_model(size, dev, ctype)
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        segments, info = model.transcribe(audio, **kwargs)
        segments = list(segments)  # el generador es perezoso: la decodificación ocurre aquí
        asr_s = time.perf_counter() - t0

    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2),
                          "p": round(w.probability, 3)})
    text = " ".join(s.text.strip() for s in segments).strip()
    ct = model.model  # objeto ctranslate2.models.Whisper: fuente de verdad del dispositivo
    return {
        "text": text,
        "words": words,
        "language": info.language,
        "model": size,
        "device_requested": device or asr_cfg["device"],
        "device": getattr(ct, "device", dev),
        "device_index": getattr(ct, "device_index", None),
        "compute_type": getattr(ct, "compute_type", ctype),  # efectiva, leída de CTranslate2
        "compute_type_requested": compute_type or asr_cfg["compute_type"],
        "cuda_devices_visible": cuda_device_count(),
        "glossary_mode": mode,
        "glossary_support": support,
        "audio_s": round(audio_s, 2),
        "asr_s": round(asr_s, 2),
        "load_s": round(load_s, 2),
        "s_per_audio_min": round(asr_s / (audio_s / 60), 2) if audio_s else None,
        "ctranslate2": ctranslate2.__version__,
        **mon.report(),
    }
