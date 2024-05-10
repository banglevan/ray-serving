from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
from ray import serve
from ray.serve.handle import DeploymentHandle
from typing import List

app = FastAPI()

@serve.deployment(num_replicas=1,
                  max_concurrent_queries=10,
                  ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class IngressText2Image:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle: DeploymentHandle = diffusion_model_handle.options(
            use_new_handle_api=True,
        )

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")