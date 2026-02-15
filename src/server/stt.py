"""
Speech-to-Text module using Whisper.

Supports:
- Remote whisper.cpp server (GPU-accelerated, fastest)
- Local faster-whisper (CTranslate2 backend)
- Local openai-whisper (fallback)
- Mock mode (testing)
"""

import asyncio
import io
import wave
from typing import Optional

import numpy as np
from loguru import logger


class WhisperSTT:
    """Whisper-based Speech-to-Text."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
        server_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.server_url = server_url.rstrip("/") if server_url else None
        self.model = None
        self._backend = "mock"
        self._http_client = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model (or connect to remote server)."""
        # Remote whisper.cpp server (highest priority)
        if self.server_url:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=30.0)
                self._backend = "remote"
                logger.info(f"Using remote whisper.cpp server: {self.server_url}")
                return
            except ImportError:
                logger.error("httpx required for remote STT server (pip install httpx)")
            except Exception as e:
                logger.warning(f"Remote STT setup failed: {e}")

        # Try faster-whisper
        try:
            from faster_whisper import WhisperModel

            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "cpu"
                    compute_type = "int8"
                else:
                    self.device = "cpu"
                    compute_type = "int8"
            elif self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"

            logger.info(f"Loading faster-whisper {self.model_name} on {self.device}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device if self.device != "mps" else "cpu",
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("✅ faster-whisper loaded")
            return
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")

        # Try openai-whisper
        try:
            import whisper

            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading openai-whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("✅ openai-whisper loaded")
            return
        except ImportError:
            logger.warning("openai-whisper not available")
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")

        # Mock mode for testing
        logger.warning("⚠️ No STT backend - using mock mode")
        self._backend = "mock"

    @staticmethod
    def _audio_to_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert float32 numpy audio to WAV bytes."""
        # Convert float32 [-1.0, 1.0] to int16
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "remote":
            return self._transcribe_remote(audio)

        elif self._backend == "faster-whisper":
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        elif self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()

        else:
            # Mock mode - return placeholder
            logger.debug(f"Mock STT: received {len(audio)} samples")
            return "[Mock transcription - install whisper for real STT]"

    def _transcribe_remote(self, audio: np.ndarray) -> str:
        """Send audio to remote whisper.cpp server for transcription."""
        wav_bytes = self._audio_to_wav(audio)

        try:
            response = self._http_client.post(
                f"{self.server_url}/inference",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "temperature": "0.0",
                    "temperature_inc": "0.2",
                    "response_format": "json",
                    "language": self.language,
                },
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("text", "").strip()
            logger.debug(f"Remote STT: {text[:80]}...")
            return text
        except Exception as e:
            logger.error(f"Remote whisper.cpp server error: {e}")
            return ""
