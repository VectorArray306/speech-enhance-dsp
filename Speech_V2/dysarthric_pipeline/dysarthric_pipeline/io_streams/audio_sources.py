"""Audio input: a WAV file, the computer microphone, or the FRDM-MCXN236 serial link."""

from __future__ import annotations

import os
import queue
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def crc16_ccitt_false(payload: bytes) -> int:
    """CRC-16/CCITT-FALSE. Initial value 0xFFFF, polynomial 0x1021."""
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_pcm_frame(samples: np.ndarray, magic: tuple[int, int] = (0xAA, 0x55)) -> bytes:
    """Pack one block of float audio into the board frame format."""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    body = (pcm * 32767.0).astype("<i2").tobytes()
    if len(body) > 65535:
        raise ValueError("Serial payload is longer than 65535 bytes.")
    header = bytes(magic) + len(body).to_bytes(2, "little")
    return header + body + crc16_ccitt_false(body).to_bytes(2, "little")


class AudioSource(ABC):
    sample_rate: int = 16000

    @abstractmethod
    def chunks(self):
        """Yield float32 mono frames at ``sample_rate``."""

    def close(self) -> None:
        return None


def resolve_input_device(sd, device: str | int | None) -> int | None:
    """Map a substring device name to a single sounddevice input index."""
    if device is None:
        return None
    if isinstance(device, int):
        return device
    text = str(device).strip()
    if text.isdigit():
        return int(text)
    needle = text.lower()
    matches: list[tuple[int, str, str]] = []
    for index, dev in enumerate(sd.query_devices()):
        if int(dev["max_input_channels"]) < 1:
            continue
        name = str(dev["name"])
        if needle not in name.lower():
            continue
        hostapi = str(sd.query_hostapis(dev["hostapi"])["name"])
        matches.append((index, name, hostapi))
    if not matches:
        return device
    if len(matches) == 1:
        return matches[0][0]

    def host_rank(hostapi: str) -> int:
        host = hostapi.lower()
        if os.name == "nt":
            if "wasapi" in host:
                return 0
            if "wdm" in host or "ks" in host:
                return 1
            if "directsound" in host:
                return 2
            if "mme" in host:
                return 3
            return 4
        if "alsa" in host:
            return 0
        if "pulse" in host or "pipewire" in host:
            return 1
        if "jack" in host:
            return 2
        return 3

    matches.sort(key=lambda item: (host_rank(item[2]), item[0]))
    chosen_index, chosen_name, chosen_host = matches[0]
    print(
        f"microphone: picked [{chosen_index}] {chosen_name} ({chosen_host})",
        flush=True,
    )
    if len(matches) > 1:
        skipped = ", ".join(f"[{idx}] {name}" for idx, name, _ in matches[1:])
        print(f"microphone: skipped other matches: {skipped}", flush=True)
    return chosen_index


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    return np.ascontiguousarray(wave)


class FileStreamSource(AudioSource):
    """Read a WAV in fixed frames. The file does not have to be 16 kHz."""

    def __init__(self, path: str, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        self.path = str(path)
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))

    def chunks(self):
        import soundfile as sf
        import torch
        import torchaudio

        audio, file_rate = sf.read(self.path, dtype="float32", always_2d=False)
        wave = _to_mono_float(audio)
        if file_rate != self.sample_rate:
            wave = torchaudio.functional.resample(torch.from_numpy(wave), file_rate, self.sample_rate).numpy()
        for start in range(0, len(wave), self.frame_samples):
            frame = wave[start : start + self.frame_samples]
            if len(frame):
                yield np.asarray(frame, dtype=np.float32)


class MicrophoneSource(AudioSource):
    """Read a microphone until the program is stopped.

    ``device`` is a sounddevice name or index. The NXP USB AUDIO DEMO
    capture opens at 48 kHz, so frames are resampled to ``sample_rate``.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, device: str | int | None = None) -> None:
        import sounddevice as sd

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.device = resolve_input_device(sd, device)
        self._stream = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()

    def _capture_rate(self, sd) -> tuple[int, dict]:
        info = sd.query_devices(self.device, "input")
        native = int(info["default_samplerate"])
        try:
            sd.check_input_settings(
                device=self.device,
                channels=1,
                dtype="float32",
                samplerate=self.sample_rate,
            )
        except Exception:
            return native, info
        return self.sample_rate, info

    def chunks(self):
        import sounddevice as sd

        capture_rate, info = self._capture_rate(sd)
        block = max(1, int(capture_rate * self.frame_ms / 1000))
        if capture_rate == self.sample_rate:
            print(f"microphone: {info['name']} at {capture_rate} Hz", flush=True)
        else:
            print(
                f"microphone: {info['name']} at {capture_rate} Hz, resampled to {self.sample_rate} Hz",
                flush=True,
            )

        def _callback(indata, frames, time_info, status):
            del frames, time_info
            if status:
                print(f"microphone: {status}", flush=True)
            self._queue.put(np.asarray(indata[:, 0], dtype=np.float32).copy())

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=capture_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=_callback,
        )
        self._stream.start()
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if capture_rate != self.sample_rate and len(frame):
                    import torch
                    import torchaudio

                    wave = torch.from_numpy(np.asarray(frame, dtype=np.float32))
                    frame = (
                        torchaudio.functional.resample(wave, capture_rate, self.sample_rate)
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                yield np.asarray(frame, dtype=np.float32)
        finally:
            self.close()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SerialSource(AudioSource):
    """Read 16 kHz mono audio from the FRDM-MCXN236 USB serial port.

    ``framed`` packets are ``0xAA 0x55 | uint16le length | int16le PCM | CRC-16``.
    ``raw`` is a plain stream of int16 samples.
    """

    def __init__(
        self,
        port: str,
        baud: int,
        mode: str = "framed",
        sample_rate: int = 16000,
        frame_ms: int = 30,
        magic: tuple[int, int] = (0xAA, 0x55),
    ) -> None:
        self.port = port
        self.baud = baud
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self.magic = bytes(magic)
        self._serial = None

    def chunks(self):
        import serial

        self._serial = serial.Serial(self.port, self.baud, timeout=0.5)
        try:
            if self.mode == "raw":
                yield from self._raw_chunks()
            elif self.mode == "framed":
                yield from self._framed_chunks()
            else:
                raise ValueError(f"Unknown serial mode {self.mode!r}. Use framed or raw.")
        finally:
            self.close()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _read_exact(self, count: int) -> bytes | None:
        data = bytearray()
        while len(data) < count:
            piece = self._serial.read(count - len(data))
            if not piece:
                return None if not data else bytes(data)
            data.extend(piece)
        return bytes(data)

    def _raw_chunks(self):
        need = self.frame_samples * 2
        while True:
            payload = self._read_exact(need)
            if not payload:
                break
            yield _pcm16_to_float(payload)

    def _framed_chunks(self):
        buffer = bytearray()
        while True:
            incoming = self._serial.read(4096)
            if not incoming:
                continue
            buffer.extend(incoming)
            while True:
                frame = _pop_frame(buffer, self.magic)
                if frame is None:
                    break
                if len(frame):
                    yield frame


def _pcm16_to_float(payload: bytes) -> np.ndarray:
    usable = payload[: len(payload) - (len(payload) % 2)]
    if not usable:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(usable, dtype="<i2").astype(np.float32) / 32768.0


def _pop_frame(buffer: bytearray, magic: bytes) -> np.ndarray | None:
    """Remove one valid frame from the front of ``buffer``. Return None if incomplete."""
    start = buffer.find(magic)
    if start < 0:
        if len(buffer) > 1:
            del buffer[:-1]
        return None
    if start:
        del buffer[:start]
    if len(buffer) < 4:
        return None
    length = int.from_bytes(buffer[2:4], "little")
    total = 4 + length + 2
    if len(buffer) < total:
        return None
    payload = bytes(buffer[4 : 4 + length])
    received = int.from_bytes(buffer[4 + length : total], "little")
    del buffer[:total]
    if length % 2 or crc16_ccitt_false(payload) != received:
        return np.zeros(0, dtype=np.float32)
    return _pcm16_to_float(payload)


def resolve_serial_port(port: str | None) -> str:
    """Pick a serial device. ``auto`` uses a USB COM port on Windows and /dev/ttyACM0 on Linux."""
    text = (port or "").strip()
    if text and text.lower() != "auto":
        return text
    if os.name == "nt":
        try:
            from serial.tools import list_ports
        except ImportError as exc:
            raise RuntimeError("Install pyserial to read the board: pip install pyserial") from exc
        ports = list(list_ports.comports())
        preferred = []
        for info in ports:
            blob = f"{info.device} {info.description} {info.hwid}".lower()
            if any(token in blob for token in ("usb", "acm", "mcx", "nxp", "serial")):
                preferred.append(info)
        chosen = preferred[0] if preferred else (ports[0] if len(ports) == 1 else None)
        if chosen is not None:
            print(f"serial: using {chosen.device} ({chosen.description})", flush=True)
            return chosen.device
        found = ", ".join(info.device for info in ports) or "none"
        raise RuntimeError(
            "Set serial.port in the config to the COM port from Device Manager. "
            f"Ports seen: {found}."
        )
    return "/dev/ttyACM0"


def frames_from_bytes(data: bytes, magic: tuple[int, int] = (0xAA, 0x55)) -> list[np.ndarray]:
    """Decode a complete framed recording. Used to check the serial format without the board."""
    buffer = bytearray(data)
    frames = []
    while buffer:
        before = len(buffer)
        frame = _pop_frame(buffer, bytes(magic))
        if frame is None or len(buffer) == before:
            break
        if len(frame):
            frames.append(frame)
    return frames
