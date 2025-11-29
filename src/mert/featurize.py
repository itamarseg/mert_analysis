from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np, torch, torchaudio, librosa
from transformers import AutoModel, Wav2Vec2FeatureExtractor

@dataclass
class MertConfig:
    model_name: str = "m-a-p/MERT-v1-330M"
    target_sr: int = 24000          # MERT v1 uses 24 kHz :contentReference[oaicite:2]{index=2}
    chunk_secs: float = 5.0         # common chunk size for MERT :contentReference[oaicite:3]{index=3}
    overlap_secs: float = 1.0
    layer: Optional[int] = -1       # which hidden_state to export; -1 = last

class MertExtractor:
    def __init__(self, cfg: MertConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True).to(self.device).eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name, trust_remote_code=True)

    def _resample_to_24k(self, y, sr):
        if sr == self.cfg.target_sr:
            return y, sr
        return librosa.resample(y, orig_sr=sr, target_sr=self.cfg.target_sr), self.cfg.target_sr

    @torch.no_grad()
    def encode(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        # resample to target SR
        if sr != self.cfg.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.target_sr)
            sr = self.cfg.target_sr

        hop = 1.0 / 75.0  # ~75 Hz frame rate for MERT v1
        step = int(self.cfg.chunk_secs * sr)
        stride = int(max(1, (self.cfg.chunk_secs - self.cfg.overlap_secs) * sr))

        feats_per_layer: Dict[int, List[np.ndarray]] = {}
        times_chunks: List[np.ndarray] = []
        pos = 0

        while pos < len(y):
            x = y[pos:pos + step]
            if len(x) < int(0.25 * sr):
                break  # ignore very short tail

            inputs = self.processor(x, sampling_rate=sr, return_tensors="pt")
            out = self.model(
                **{k: v.to(self.device) for k, v in inputs.items()},
                output_hidden_states=True
            )
            hs = out.hidden_states  # tuple: [emb, L1, ..., L_N]

            # Decide which layers to keep
            if self.cfg.layer == "all":
                keep = list(range(len(hs)))  # include the input embedding at index 0
            elif isinstance(self.cfg.layer, list):
                keep = [(i if i >= 0 else len(hs) + i) for i in self.cfg.layer]
            else:
                # single int
                i = self.cfg.layer if self.cfg.layer >= 0 else len(hs) + self.cfg.layer
                keep = [i]

            # Time grid for this chunk (match the length of the last layer)
            t0 = pos / sr
            T_chunk = hs[-1].shape[1]
            t_chunk = t0 + np.arange(T_chunk) * hop
            times_chunks.append(t_chunk)

            # Accumulate features
            for li in keep:
                h = hs[li].squeeze(0).cpu().numpy()  # [T,D]
                feats_per_layer.setdefault(li, []).append(h)

            pos += stride

        times = np.concatenate(times_chunks) if times_chunks else np.zeros((0,), np.float32)

        out_pack: Dict[str, np.ndarray] = {"times": times.astype(np.float32), "sr": np.array(sr, dtype=np.int32)}
        for li, chunks in feats_per_layer.items():
            H = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, hs[li].shape[-1]), np.float32)
            # name like emb_L-1, emb_L12, emb_L0
            key = f"emb_L{li if li < len(hs) else li}"
            out_pack[key] = H
        return out_pack
