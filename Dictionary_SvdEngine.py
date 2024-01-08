#
# This is based on scripts/demo/video_sampling.py but removes the dependency on
# the Streamlit library and UI.
#
# Created: 8 January 2024
#

import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import seed_everything

import Dictionary_SvdEngineHelper

lowvram_mode = True

SAVE_PATH = "outputs/demo/vid/"

# 'svd' from VERSION2SPECS in video_sampling.py
version_dict = {
        "T": 50, # modified from original value of 14
        "H": 64 * 4, # modified from original value of 576 (64*9)
        "W": 64 * 4, # modified from original value of 1024 (64*16)
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd.safetensors",
        "options": {
            "discretization": "EDMDiscretization",
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": "LinearPredictionGuider",
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 20, # modified from original value of 25
            "decoding_t": 10 # modified from original value of T
        },
    }

class Dictionary_SvdEngine:

    helper = None

    model = None
    sampler = None

    options = None

    save_locally = None
    save_path = None

    value_dict = None
    version = "svd"

    H = W = T = C = F = None

    def __init__(self):
        self.helper = Dictionary_SvdEngineHelper.Dictionary_SvdEngineHelper()

        self.H = version_dict["H"]
        self.W = version_dict["W"]
        self.T = version_dict["T"]
        self.C = version_dict["C"]
        self.F = version_dict["f"]
        self.options = version_dict["options"]

        state = self.helper.init_state(version_dict, load_filter=True)
        if state["msg"]:
            print(state["msg"])
        self.model = state["model"]

        ukeys = set(
            self.helper.get_unique_embedder_keys_from_conditioner(state["model"].conditioner)
        )

        self.value_dict = self.helper.init_embedder_options(
            ukeys,
            {},
        )

        self.value_dict["image_only_indicator"] = 0

        self.save_locally, self.save_path = self.helper.init_save_locally(
            os.path.join(SAVE_PATH, self.version), init_value=True
        )

        self.options["num_frames"] = self.T

        self.sampler, num_rows, num_cols = self.helper.init_sampling(options=self.options)
        self.num_samples = num_rows * num_cols

    def generate_video(self, image_bytes: bytearray) -> bytearray:

        seed = random.randint(1, np.iinfo(np.uint32).max) #23
        seed_everything(seed)

        img = self.helper.load_img_for_prediction(self.W, self.H, image_bytes = image_bytes)
        cond_aug = 0.02

        self.value_dict["cond_frames_without_noise"] = img
        self.value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
        self.value_dict["cond_aug"] = cond_aug

        decoding_t = self.options.get("decoding_t", self.T)

        out = self.helper.do_sample(
                self.model,
                self.sampler,
                self.value_dict,
                self.num_samples,
                self.H,
                self.W,
                self.C,
                self.F,
                T=self.T,
                batch2model_input=["num_video_frames", "image_only_indicator"],
                force_uc_zero_embeddings=self.options.get("force_uc_zero_embeddings", None),
                force_cond_zero_embeddings=self.options.get(
                    "force_cond_zero_embeddings", None
                ),
                return_latents=False,
                decoding_t=decoding_t,
            )

        if isinstance(out, (tuple, list)):
            samples, _ignore = out
        else:
            samples = out

        if self.save_locally:
            saving_fps = self.value_dict["fps"]
            video_bytes = self._save_video_as_grid_and_mp4(samples, self.save_path, self.T, fps=saving_fps)
            return video_bytes


    def _save_video_as_grid_and_mp4(self,
            video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5
        ) -> bytearray:
            os.makedirs(save_path, exist_ok=True)
            base_count = len(glob(os.path.join(save_path, "*.mp4")))

            video_batch = rearrange(video_batch, "(b t) c h w -> b t c h w", t=T)
            # video_batch = embed_watermark(video_batch)
            for vid in video_batch:
                # save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)

                video_path = os.path.join(save_path, f"{base_count:06d}.mp4")

                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (vid.shape[-1], vid.shape[-2]),
                )
                try:
                    vid = (
                        (rearrange(vid, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
                    )
                    for frame in vid:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame)
                finally:
                    writer.release()

                # video_path_h264 = video_path[:-4] + "_h264.mp4"
                # os.system(f"ffmpeg -i {video_path} -c:v libx264 {video_path_h264}")

                with open(video_path, "rb") as f:
                    video_bytes = f.read()

                base_count += 1

                return video_bytes
