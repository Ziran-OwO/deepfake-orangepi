import torch
import librosa
import numpy as np
import os
from LEMAS import LEMAS
import python_speech_features as ps

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
            "frame_logits": frame_logits_np,  # 帧级logits 每一行是[fake_logit, real_logit]
            "projected_vector": projected_vector_np  # 128维投影向量
        }


if __name__ == "__main__":
    PTH_MODEL_PATH = "./e20_tloss0.0023_dloss0.0051_deer0.1570.pth"
    TEST1_DIR = "./real"

    inferencer = LEMASLogitsInferencer(PTH_MODEL_PATH, device="cpu")

    audio_extensions = (".flac", ".wav")
    for filename in os.listdir(TEST1_DIR):
        if filename.lower().endswith(audio_extensions):
            audio_path = os.path.join(TEST1_DIR, filename)
            print("=" * 80)
            print(f"音频文件: {filename}")
            print("=" * 80)

            results = inferencer.get_logits(audio_path)

            print(f"\n【全局Logits】")
            print(f"伪造(fake) logit: {results['global_logits'][0]:.6f}")
            print(f"真实(real) logit: {results['global_logits'][1]:.6f}")

            # 打印帧级logits（展示前5帧和后5帧）
            print(f"\n【帧级Logits】（共{len(results['frame_logits'])}帧）")
            print("前5帧logits:")
            for i in range(min(5, len(results['frame_logits']))):
                print(
                    f"  帧{i + 1}: fake={results['frame_logits'][i][0]:.6f}, real={results['frame_logits'][i][1]:.6f}")
            if len(results['frame_logits']) > 5:
                print("后5帧logits:")
                for i in range(-5, 0):
                    print(
                        f"  帧{len(results['frame_logits']) + i + 1}: fake={results['frame_logits'][i][0]:.6f}, real={results['frame_logits'][i][1]:.6f}")

            # 打印projected_vector（128维投影向量）
            print(f"\n【Projected Vector】（shape: {results['projected_vector'].shape}）")
            print(f"前10个元素: {results['projected_vector'][:10].round(6)}")
            print(f"后10个元素: {results['projected_vector'][-10:].round(6)}")
            print(
                f"向量均值: {np.mean(results['projected_vector']):.6f}, 标准差: {np.std(results['projected_vector']):.6f}")
            print("\n")