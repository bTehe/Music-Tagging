"""Log-mel feature frontend."""

from __future__ import annotations

import numpy as np


class LogMelFrontend:
    """Compute log-mel spectrograms using torchaudio when available."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 96,
        f_min: float = 0.0,
        f_max: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self._torchaudio_ready = False
        try:
            import torchaudio

            self._transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                center=True,
                power=2.0,
            )
            self._torchaudio_ready = True
        except Exception:
            self._torchaudio_ready = False

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self._torchaudio_ready:
            return self._with_torchaudio(audio)
        return self._with_librosa(audio)

    def _with_torchaudio(self, audio: np.ndarray) -> np.ndarray:
        import torch

        wav = torch.from_numpy(audio).float().unsqueeze(0)
        mel = self._transform(wav).squeeze(0)
        logmel = torch.log(mel + self.eps)
        return logmel.cpu().numpy().astype(np.float32)

    def _with_librosa(self, audio: np.ndarray) -> np.ndarray:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )
        return np.log(mel + self.eps).astype(np.float32)
