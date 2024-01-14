#
# This is based on scripts/demo/streamlit_helpers.py but removes the dependency
# on the Streamlit library and UI.
#
# Created: 8 January 2024
#

import copy
import io
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as TT
from einops import repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms

from scripts.demo.discretization import (Img2ImgDiscretizationWrapper,
                                         Txt2NoisyDiscretizationWrapper)
from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.modules.diffusionmodules.sampling import (DPMPP2MSampler,
                                                   DPMPP2SAncestralSampler,
                                                   EulerAncestralSampler,
                                                   EulerEDMSampler,
                                                   HeunEDMSampler,
                                                   LinearMultistepSampler)
from sgm.util import instantiate_from_config

import math
import os
from typing import List, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import autocast

from sgm.modules.diffusionmodules.guiders import (LinearPredictionGuider,
                                                  VanillaCFG)
from sgm.util import default

lowvram_mode = True

class Dictionary_SvdEngineHelper:

    def do_sample(self,
        model,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings: Optional[List] = None,
        force_cond_zero_embeddings: Optional[List] = None,
        batch2model_input: List = None,
        return_latents=False,
        filter=None,
        T=None,
        additional_batch_uc_fields=None,
        decoding_t=None,
    ):
        force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
        batch2model_input = default(batch2model_input, [])
        additional_batch_uc_fields = default(additional_batch_uc_fields, [])

        print("Sampling")

        outputs = None
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    if T is not None:
                        num_samples = [num_samples, T]
                    else:
                        num_samples = [num_samples]

                    self.load_model(model.conditioner)
                    batch = batch_uc = c = uc = None
                    try:
                        batch, batch_uc = self.get_batch(
                            self.get_unique_embedder_keys_from_conditioner(model.conditioner),
                            value_dict,
                            num_samples,
                            T=T,
                            additional_batch_uc_fields=additional_batch_uc_fields,
                        )

                        c, uc = model.conditioner.get_unconditional_conditioning(
                            batch,
                            batch_uc=batch_uc,
                            force_uc_zero_embeddings=force_uc_zero_embeddings,
                            force_cond_zero_embeddings=force_cond_zero_embeddings,
                        )
                    finally:
                        self.unload_model(model.conditioner)

                    for k in c:
                        if not k == "crossattn":
                            c[k], uc[k] = map(
                                lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                            )
                        if k in ["crossattn", "concat"] and T is not None:
                            uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                            c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                    additional_model_inputs = {}
                    for k in batch2model_input:
                        if k == "image_only_indicator":
                            assert T is not None

                            if isinstance(
                                sampler.guider, (VanillaCFG, LinearPredictionGuider)
                            ):
                                additional_model_inputs[k] = torch.zeros(
                                    num_samples[0] * 2, num_samples[1]
                                ).to("cuda")
                            else:
                                additional_model_inputs[k] = torch.zeros(num_samples).to(
                                    "cuda"
                                )
                        else:
                            additional_model_inputs[k] = batch[k]

                    print(additional_model_inputs)

                    shape = (math.prod(num_samples), C, H // F, W // F)

                    self.load_model(model.denoiser)
                    try:
                        self.load_model(model.model)
                        try:
                            def denoiser(input, sigma, c):
                                return model.denoiser(
                                    model.model, input, sigma, c, **additional_model_inputs
                                )
                            randn = torch.randn(shape).to("cuda")
                            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                        finally:
                            self.unload_model(model.model)
                    finally:
                        self.unload_model(model.denoiser)

                    self.load_model(model.first_stage_model)
                    try:
                        model.en_and_decode_n_samples_a_time = (
                            decoding_t  # Decode n frames at a time
                        )
                        samples_x = model.decode_first_stage(samples_z)
                        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    finally:
                        self.unload_model(model.first_stage_model)

                    if filter is not None:
                        samples = filter(samples)

                    # if T is None:
                    #     grid = torch.stack([samples])
                    #     grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                    #     outputs.image(grid.cpu().numpy())
                    # else:
                    #     as_vids = rearrange(samples, "(b t) c h w -> b t c h w", t=T)
                    #     for i, vid in enumerate(as_vids):
                    #         grid = rearrange(make_grid(vid, nrow=4), "c h w -> h w c")
                    #         # TODO Create non-Streamlit representation of this
                    #         # st.image(
                    #         #     grid.cpu().numpy(),
                    #         #     f"Sample #{i} as image",
                    #         # )

                    if return_latents:
                        return samples, samples_z
                    return samples

    def get_discretization(self, discretization, options, key=1):
        if discretization == "LegacyDDPMDiscretization":
            discretization_config = {
                "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
            }
        elif discretization == "EDMDiscretization":
            sigma_min = options.get("sigma_min", 0.03)
            sigma_max = options.get("sigma_max", 14.61)
            rho = options.get("rho", 3.0)
            discretization_config = {
                "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
                "params": {
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "rho": rho,
                },
            }

        return discretization_config

    def get_guider(self, options, key):
        guider = options.get("guider", "VanillaCFG")

        additional_guider_kwargs = options.pop("additional_guider_kwargs", {})

        if guider == "IdentityGuider":
            guider_config = {
                "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
            }
        elif guider == "VanillaCFG":
            scale = options.get("cfg", 5.0)

            guider_config = {
                "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
                "params": {
                    "scale": scale,
                    **additional_guider_kwargs,
                },
            }
        elif guider == "LinearPredictionGuider":
            max_scale = options.get("cfg", 1.5)
            min_scale = options.get("min_cfg", 1.0)

            guider_config = {
                "target": "sgm.modules.diffusionmodules.guiders.LinearPredictionGuider",
                "params": {
                    "max_scale": max_scale,
                    "min_scale": min_scale,
                    "num_frames": options["num_frames"],
                    **additional_guider_kwargs,
                },
            }
        else:
            raise NotImplementedError
        return guider_config

    def get_resizing_factor(self,
        desired_shape: Tuple[int, int], current_shape: Tuple[int, int]
    ) -> float:
        r_bound = desired_shape[1] / desired_shape[0]
        aspect_r = current_shape[1] / current_shape[0]
        if r_bound >= 1.0:
            if aspect_r >= r_bound:
                factor = min(desired_shape) / min(current_shape)
            else:
                if aspect_r < 1.0:
                    factor = max(desired_shape) / min(current_shape)
                else:
                    factor = max(desired_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = min(desired_shape) / min(current_shape)
            else:
                if aspect_r > 1:
                    factor = max(desired_shape) / min(current_shape)
                else:
                    factor = max(desired_shape) / max(current_shape)

        return factor

    def get_sampler(self, sampler_name, steps, discretization_config, guider_config, key=1):
        if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
            s_churn = 0.0
            s_tmin = 0.0
            s_tmax = 999.0
            s_noise = 1.0

            if sampler_name == "EulerEDMSampler":
                sampler = EulerEDMSampler(
                    num_steps=steps,
                    discretization_config=discretization_config,
                    guider_config=guider_config,
                    s_churn=s_churn,
                    s_tmin=s_tmin,
                    s_tmax=s_tmax,
                    s_noise=s_noise,
                    verbose=True,
                )
            elif sampler_name == "HeunEDMSampler":
                sampler = HeunEDMSampler(
                    num_steps=steps,
                    discretization_config=discretization_config,
                    guider_config=guider_config,
                    s_churn=s_churn,
                    s_tmin=s_tmin,
                    s_tmax=s_tmax,
                    s_noise=s_noise,
                    verbose=True,
                )
        elif (
            sampler_name == "EulerAncestralSampler"
            or sampler_name == "DPMPP2SAncestralSampler"
        ):
            s_noise = 1.0
            eta = 1.0

            if sampler_name == "EulerAncestralSampler":
                sampler = EulerAncestralSampler(
                    num_steps=steps,
                    discretization_config=discretization_config,
                    guider_config=guider_config,
                    eta=eta,
                    s_noise=s_noise,
                    verbose=True,
                )
            elif sampler_name == "DPMPP2SAncestralSampler":
                sampler = DPMPP2SAncestralSampler(
                    num_steps=steps,
                    discretization_config=discretization_config,
                    guider_config=guider_config,
                    eta=eta,
                    s_noise=s_noise,
                    verbose=True,
                )
        elif sampler_name == "DPMPP2MSampler":
            sampler = DPMPP2MSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                verbose=True,
            )
        elif sampler_name == "LinearMultistepSampler":
            order = 4
            sampler = LinearMultistepSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                order=order,
                verbose=True,
            )
        else:
            raise ValueError(f"unknown sampler {sampler_name}!")

        return sampler

    def init_sampling(self,
        key=1,
        img2img_strength: Optional[float] = None,
        specify_num_samples: bool = True,
        stage2strength: Optional[float] = None,
        options: Optional[Dict[str, int]] = None,
        default_steps = 40
    ):
        options = {} if options is None else options

        num_rows, num_cols = 1, 1
        if specify_num_samples:
            num_cols = num_cols

        steps = options.get("num_steps", default_steps)
        sampler = options.get("sampler", "EulerEDMSampler")
        discretization = options.get("discretization", "LegacyDDPMDiscretization")

        discretization_config = self.get_discretization(discretization, options=options, key=key)

        guider_config = self.get_guider(options=options, key=key)

        sampler = self.get_sampler(sampler, steps, discretization_config, guider_config, key=key)
        if img2img_strength is not None:
            print(
                f"WARNING: Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper"
            )
            sampler.discretization = Img2ImgDiscretizationWrapper(
                sampler.discretization, strength=img2img_strength
            )
        if stage2strength is not None:
            sampler.discretization = Txt2NoisyDiscretizationWrapper(
                sampler.discretization, strength=stage2strength, original_steps=steps
            )
        return sampler, num_rows, num_cols

    def get_unique_embedder_keys_from_conditioner(self, conditioner):
        return list(set([x.input_key for x in conditioner.embedders]))


    def get_batch(self,
        keys,
        value_dict: dict,
        N: Union[List, ListConfig],
        device: str = "cuda",
        T: int = None,
        additional_batch_uc_fields: List[str] = [],
    ):
        # Hardcoded demo setups; might undergo some changes in the future

        batch = {}
        batch_uc = {}

        for key in keys:
            if key == "txt":
                batch["txt"] = [value_dict["prompt"]] * math.prod(N)

                batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

            elif key == "original_size_as_tuple":
                batch["original_size_as_tuple"] = (
                    torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
            elif key == "crop_coords_top_left":
                batch["crop_coords_top_left"] = (
                    torch.tensor(
                        [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                    )
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
            elif key == "aesthetic_score":
                batch["aesthetic_score"] = (
                    torch.tensor([value_dict["aesthetic_score"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
                batch_uc["aesthetic_score"] = (
                    torch.tensor([value_dict["negative_aesthetic_score"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )

            elif key == "target_size_as_tuple":
                batch["target_size_as_tuple"] = (
                    torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
            elif key == "fps":
                batch[key] = (
                    torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
                )
            elif key == "fps_id":
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
                )
            elif key == "motion_bucket_id":
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .to(device)
                    .repeat(math.prod(N))
                )
            elif key == "pool_image":
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                    device, dtype=torch.half
                )
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames":
                batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
            elif key == "cond_frames_without_noise":
                batch[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
                )
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
            elif key in additional_batch_uc_fields and key not in batch_uc:
                batch_uc[key] = copy.copy(batch[key])
        return batch, batch_uc

    def init_embedder_options(self, keys, init_dict, prompt=None, negative_prompt=None):
        value_dict = {}
        for key in keys:
            if key == "txt":
                if prompt is None:
                    prompt = "A professional photograph of an astronaut riding a pig"
                if negative_prompt is None:
                    negative_prompt = ""

                value_dict["prompt"] = prompt
                value_dict["negative_prompt"] = negative_prompt

            if key == "original_size_as_tuple":
                orig_width = init_dict["orig_width"]
                orig_height = init_dict["orig_height"]

                value_dict["orig_width"] = orig_width
                value_dict["orig_height"] = orig_height

            if key == "crop_coords_top_left":
                crop_coord_top = 0
                crop_coord_left = 0

                value_dict["crop_coords_top"] = crop_coord_top
                value_dict["crop_coords_left"] = crop_coord_left

            if key == "aesthetic_score":
                value_dict["aesthetic_score"] = 6.0
                value_dict["negative_aesthetic_score"] = 2.5

            if key == "target_size_as_tuple":
                value_dict["target_width"] = init_dict["target_width"]
                value_dict["target_height"] = init_dict["target_height"]

            if key in ["fps_id", "fps"]:
                fps = 25 #6

                value_dict["fps"] = fps
                value_dict["fps_id"] = fps - 1

            if key == "motion_bucket_id":
                mb_id = 127
                value_dict["motion_bucket_id"] = mb_id

            if key == "pool_image":
                image = self.load_img(
                    key="pool_image_input",
                    size=224,
                    center_crop=True,
                )
                if image is None:
                    print("Need an image here")
                    image = torch.zeros(1, 3, 224, 224)
                value_dict["pool_image"] = image

        return value_dict

    def initial_model_load(self, model):
        global lowvram_mode
        if lowvram_mode:
            model.model.half()
        else:
            model.cuda()
        return model

    def load_img(self,
        size: Union[None, int, Tuple[int, int]] = None,
        center_crop: bool = False,
        image = None
    ):
        if image is None:
            return None
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")

        transform = []
        if size is not None:
            transform.append(transforms.Resize(size))
        if center_crop:
            transform.append(transforms.CenterCrop(size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Lambda(lambda x: 2.0 * x - 1.0))

        transform = transforms.Compose(transform)
        img = transform(image)[None, ...]
        print(f"input min/max/mean: {img.min():.3f}/{img.max():.3f}/{img.mean():.3f}")
        return img

    def load_model(self, model):
        model.cuda()

    def load_model_from_config(self, config, ckpt=None, verbose=True):
        model = instantiate_from_config(config.model)

        if ckpt is not None:
            print(f"Loading model from {ckpt}")
            if ckpt.endswith("ckpt"):
                pl_sd = torch.load(ckpt, map_location="cpu")
                if "global_step" in pl_sd:
                    global_step = pl_sd["global_step"]
                    print(f"loaded ckpt from global step {global_step}")
                    print(f"Global Step: {pl_sd['global_step']}")
                sd = pl_sd["state_dict"]
            elif ckpt.endswith("safetensors"):
                sd = load_safetensors(ckpt)
            else:
                raise NotImplementedError

            msg = None

            m, u = model.load_state_dict(sd, strict=False)

            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
        else:
            msg = None

        model = self.initial_model_load(model)
        model.eval()
        return model, msg

    def init_save_locally(self, _dir, init_value: bool = False):
        save_locally = init_value
        if save_locally:
            save_path = os.path.join(_dir, "samples")
        else:
            save_path = None

        return save_locally, save_path

    def init_state(self, version_dict, load_ckpt=True, load_filter=True):
        state = dict()
        if not "model" in state:
            config = version_dict["config"]
            ckpt = version_dict["ckpt"]

            config = OmegaConf.load(config)
            model, msg = self.load_model_from_config(config, ckpt if load_ckpt else None)

            state["msg"] = msg
            state["model"] = model
            state["ckpt"] = ckpt if load_ckpt else None
            state["config"] = config
            if load_filter:
                state["filter"] = DeepFloydDataFiltering(verbose=False)
        return state

    def load_img_for_prediction(self,
        W: int, H: int, key=None, device="cuda", image_bytes: bytearray = None
    ) -> torch.Tensor:
        if image_bytes is None:
            return None

        # Convert the image data to an Image object
        image = Image.open(io.BytesIO(image_bytes))
        w, h = image.size

        image = np.array(image).transpose(2, 0, 1)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
        image = image.unsqueeze(0)

        rfs = self.get_resizing_factor((H, W), (h, w))
        resize_size = [int(np.ceil(rfs * s)) for s in (h, w)]
        top = (resize_size[0] - H) // 2
        left = (resize_size[1] - W) // 2

        image = torch.nn.functional.interpolate(
            image, resize_size, mode="area", antialias=False
        )
        image = TT.functional.crop(image, top=top, left=left, height=H, width=W)

        return image.to(device) * 2.0 - 1.0

    def unload_model(self, model):
        global lowvram_mode
        if lowvram_mode:
            model.cpu()
            torch.cuda.empty_cache()
