#
# Created: 14 January 2024
#

import io
import os
import random

import numpy as np
from einops import rearrange
from PIL import Image
from pytorch_lightning import seed_everything

import Dictionary_SvdEngineHelper

lowvram_mode = True

SAVE_PATH = "outputs/demo/text2img/"

# 'SDXL-base-1.0' from VERSION2SPECS in sampling.py
version_dict = {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    }

class Dictionary_SdxlEngine:

    helper = None

    model = None
    sampler = None

    options = None

    save_locally = None
    save_path = None

    state = None

    value_dict = None

    version = 'SDXL-base-1.0'

    def __init__(self):
        self.helper = Dictionary_SvdEngineHelper.Dictionary_SvdEngineHelper()

        self.state = self.helper.init_state(version_dict, load_filter=False)
        if self.state["msg"]:
            print(self.state["msg"])
        self.model = self.state["model"]

    def generate_image(self, prompt, steps = 40, seed = None):

        if not seed:
            seed = random.randint(1, np.iinfo(np.uint32).max)
        seed_everything(seed)

        self.save_locally, self.save_path = self.helper.init_save_locally(
            os.path.join(SAVE_PATH, self.version), init_value= True
        )

        if self.version.startswith("SDXL-base"):
            add_pipeline = False
        else:
            add_pipeline = False

        is_legacy = version_dict["is_legacy"]

        if is_legacy:
            negative_prompt = "" #st.text_input("negative prompt", "")
        else:
            negative_prompt = ""  # which is unused

        stage2strength = None
        finish_denoising = False

        if add_pipeline:
            stage2strength = 0.0

            finish_denoising = True
            if not finish_denoising:
                stage2strength = None

        out = self._run_txt2img(prompt,
            self.state,
            self.version,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=self.state.get("filter"),
            stage2strength=stage2strength,
            steps = steps
        )

        if isinstance(out, (tuple, list)):
            samples, samples_z = out
        else:
            samples = out
            samples_z = None

        if samples is not None:
            # Use first sample to get image data
            sample = samples[0]
            sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")

            # Convert the tensor to an in-memory PNG image and return
            img = Image.fromarray(sample.astype(np.uint8))
            with io.BytesIO() as img_data:
                img.save(img_data, format = 'PNG')
                return img_data.getvalue()

    def _run_txt2img(self,
        prompt,
        state,
        version,
        version_dict,
        is_legacy=False,
        return_latents=False,
        filter=None,
        stage2strength=None,
        steps = 40
    ):
        H = version_dict["H"]
        W = version_dict["W"]
        C = version_dict["C"]
        F = version_dict["f"]

        init_dict = {
            "orig_width": W,
            "orig_height": H,
            "target_width": W,
            "target_height": H,
        }
        value_dict = self.helper.init_embedder_options(
            self.helper.get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
            init_dict,
            prompt=prompt,
            negative_prompt="",
        )
        sampler, num_rows, num_cols = self.helper.init_sampling(
            stage2strength = stage2strength,
            default_steps = steps)
        num_samples = num_rows * num_cols

        print(f"**Model I:** {version}")
        out = self.helper.do_sample(
            state["model"],
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=return_latents,
            filter=filter,
        )
        return out
