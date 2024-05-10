from io import BytesIO
import torch
from ray import serve
from config.config import conf

conf_txt2image = conf['txt2image']
MODEL_ID = conf_txt2image['model']['_id']
NUM_GPUS = conf_txt2image['processor']['num_gpus']
MIN_REPS = conf_txt2image['processor']['min_rep']
MAX_REPS = conf_txt2image['processor']['max_rep']
REVISION = conf_txt2image['inference']['revision']
IMAGE_SZ = conf_txt2image['inference']['image_size']

@serve.deployment(
    ray_actor_options={"num_gpus": NUM_GPUS},
    autoscaling_config={"min_replicas": MIN_REPS,
                        "max_replicas": MAX_REPS},
)
class StableDiffusionV2:
    def __init__(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
        model_id = MODEL_ID
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id,
                                                           subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                            scheduler=scheduler,
                                                            revision=REVISION,
                                                            torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = IMAGE_SZ):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image