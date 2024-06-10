"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torchaudio
import torchaudio.transforms as transforms
from moviepy.editor import VideoFileClip
from omegaconf import OmegaConf
import torchaudio.compliance.kaldi as ta_kaldi

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.models.beats.Tokenizers import TokenizersConfig, Tokenizers

MAX_INT = registry.get("MAX_INT")


@registry.register_processor("beats_audio")
class BeatsAudioProcessor(BaseProcessor):
    def __init__(self, model_name, sampling_rate, n_frames, frame_length, is_eval):
        """
        Adapted from https://github.com/NINAnor/rare_species_detections/blob/main/BEATs/BEATs.py
        """
        super().__init__()

        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.n_frames = n_frames
        self.frame_length = frame_length
        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582
        self.is_eval = is_eval

    def _load_audio(self, aupath):
        if aupath.endswith('.mp4'):
            video = VideoFileClip(aupath)
            audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
            if len(audio_np.shape) == 2:
                audio_np = audio_np.mean(axis=1)  # Convert to mono
            waveform = torch.tensor(audio_np).float()
            sr = self.sampling_rate
        else:
            waveform, sr = torchaudio.load(aupath)
            if waveform.shape[0] == 2: 
                waveform = torch.mean(waveform, dim=0)
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)
        return waveform

    def __call__(self, aupath, start_sec=None, end_sec=None):
        """
        Args:
            aupath: path to audio file
        Returns:
            torch.tensor: audio clip after transforms.
        """
        # Helper function to return empty tensor for invalid audio
        def empty_audio_tensor():
            return torch.zeros((self.n_frames, self.frame_length, 128))
        
        try:
            # Handle MP4 files
            if aupath.endswith('.mp4'):
                video = VideoFileClip(aupath)
                if start_sec is not None and end_sec is not None:
                    video = video.subclip(start_sec, end_sec)
                audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)  # Convert to mono
                waveform = torch.tensor(audio_np).float()
                sr = self.sampling_rate
            else:
                waveform, sr = torchaudio.load(aupath)

            # Validate waveform
            if len(waveform.shape) == 0:
                return empty_audio_tensor()

            # Convert stereo to mono
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0)

            # Resample waveform if necessary
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)

        except:
            return empty_audio_tensor()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform * 2**15

        # Compute fbank features
        try:
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=self.sampling_rate,
                frame_length=25,
                frame_shift=10,
            )
            fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        except:
            return empty_audio_tensor()

        # Handle padding and frames extraction differently for eval and training modes
        if not self.is_eval:
            fbank_pad_len = self.frame_length * self.n_frames - fbank.shape[0]
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            fbank = fbank[:self.frame_length * self.n_frames]
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(self.n_frames)]
        else:
            fbank_pad_len = fbank.shape[0] % self.frame_length
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            curr_frames = fbank.shape[0] // self.frame_length
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(curr_frames)]

        return torch.cat(frames, dim=0)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        return cls(
            model_name=cfg.get("model_name", 'iter3'),
            sampling_rate=cfg.get("sampling_rate", 16000),
            n_frames=cfg.get("n_frames", 2),
            frame_length=cfg.get("frame_length", 512),
            is_eval=cfg.get("is_eval", False)
        )