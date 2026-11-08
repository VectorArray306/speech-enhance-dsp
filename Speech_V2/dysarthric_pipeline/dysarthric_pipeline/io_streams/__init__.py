"""Live audio input, speech detection, and preset voice output."""

from dysarthric_pipeline.io_streams.audio_sources import (
    AudioSource,
    FileStreamSource,
    MicrophoneSource,
    SerialSource,
)

__all__ = [
    "AudioSource",
    "FileStreamSource",
    "MicrophoneSource",
    "SerialSource",
]
