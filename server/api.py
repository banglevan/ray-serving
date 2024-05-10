from server.route_txt2img import IngressText2Image
from src.txt2img import StableDiffusionV2
from ray import serve
import ray


entrypoint = IngressText2Image.bind(StableDiffusionV2.bind())
ray.init()
serve.start(detached=True, http_options={'host': '127.0.0.1', 'port': 8000})
# serve.run(entrypoint)