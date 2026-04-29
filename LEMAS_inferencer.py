import torch
import librosa
import numpy as np
import os
import python_speech_features as ps
from LEMAS import LEMAS

MAX_LEN = 64600
SR = 16000
MEL_BEGIN = 0
MEL_END = 300
NFILT = 40

class LEMASLogitsInferencer:
    def __init__(self, pth_ckpt_path, device="cpu"):
        self.device = torch.device(device)
        self.model = LEMAS().to(self.device)
        ckpt = torch.load(pth_ckpt_path, map_location=self.device)
        load_dict = ckpt if not isinstance(ckpt, dict) else ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(load_dict, strict=False)
        self.model.eval()

    def preprocess(self, audio_path):
        wave, _ = librosa.load(audio_path, sr=SR)
        wave = np.array(wave, dtype=float)

        x_len = wave.shape[0]
        if x_len >= MAX_LEN:
            wave = wave[:MAX_LEN]
        else:
            num_repeats = int(MAX_LEN / x_len) + 1
            wave = np.tile(wave, (1, num_repeats))[:, :MAX_LEN][0]

        mel_spec = ps.logfbank(wave, SR, nfilt=NFILT)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)
        spec = np.stack([
            mel_spec[MEL_BEGIN:MEL_END, :],
            delta1[MEL_BEGIN:MEL_END, :],
            delta2[MEL_BEGIN:MEL_END, :]
        ], axis=0)

        wave_tensor = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).to(self.device)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(self.device)
        freq_aug = torch.tensor([False], dtype=torch.bool).to(self.device)

        return wave_tensor, spec_tensor, freq_aug

    def get_logits(self, audio_path):
        wave, spec, freq_aug = self.preprocess(audio_path)

        with torch.no_grad():
            global_logits, frame_logits, projected_vector = self.model(wave, spec, freq_aug)

        global_logits_np = global_logits.squeeze(0).cpu().numpy()  # (2,)
        frame_logits_np = frame_logits.squeeze(0).cpu().numpy()  # (T, 2)
        projected_vector_np = projected_vector.squeeze(0).cpu().numpy()  # (128,)

        return {
            "global_logits": global_logits_np,  # 全局logits [fake_logit, real_logit]
            "frame_logits": frame_logits_np,    # 帧级logits 每一行是[fake_logit, real_logit]
            "projected_vector": projected_vector_np  # 128维投影向量
        }