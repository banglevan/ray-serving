from typing import List
from PIL import Image
import requests
import starlette.requests
import cv2
import numpy as np
from fastapi.responses import Response
import ray
from fastapi import FastAPI
from ray import serve

app = FastAPI()
@serve.deployment(num_replicas=1,
                  max_concurrent_queries=10,
                  ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class Detector:
    def __init__(self):
        pass

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=2)
    async def handle_batch(self, requests: List):
        results = []
        for request in requests:
            data = await request.form()
            infor = data._dict['data3']
            bimage = await data._dict['file1'].read()
            npimg = np.frombuffer(bimage, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pimage = Image.fromarray(frame)
            print(len(requests))
        return ['x']

    @app.post(
        "/image_rotate",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def upload(self, request: starlette.requests.Request):
        return await self.handle_batch(request)
deploy = Detector.bind()
ray.init()
serve.start(detached=True, http_options={"host":'0.0.0.0', "port": 9982})
# serve.run(deploy)
#
# path = r"C:\BANGLV\ray_serving\data\output.png"
# resp=requests.post(url="http://127.0.0.1:9982/image_rotate",
#                    data={"data1": '1', "data3": '2', "data2": '3'},
#                    files={'file1': open(path, 'rb'),
#                           'file2': open(path, 'rb')})