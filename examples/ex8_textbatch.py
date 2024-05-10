from typing import List

from starlette.requests import Request
from transformers import pipeline

from ray import serve

@serve.deployment
class BatchTextGenerator:
    def __init__(self, pipeline_key: str, model_key: str):
        self.model = pipeline(pipeline_key, model_key)

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=2)
    async def handle_batch(self, inputs: List[str]) -> List[str]:
        print("Our input array has length:", len(inputs))
        results = self.model(inputs)
        return [result[0]["generated_text"] for result in results]

    async def __call__(self, request: Request) -> List[str]:
        return await self.handle_batch(request.query_params["text"])

generator = BatchTextGenerator.bind("text-generation", "gpt2")
from ray.serve.handle import DeploymentHandle
handle: DeploymentHandle = serve.run(generator).options(use_new_handle_api=True)

input_batch = [
    'Once upon a time,',
    'Hi my name is Lewis and I like to',
    'My name is Mary, and my favorite',
    'My name is Clara and I am',
    'My name is Julien and I like to',
    'Today I accidentally',
    'My greatest wish is to',
    'In a galaxy far far away',
    'My best talent is',
]
print("Input batch is", input_batch)

import ray
responses = [handle.handle_batch.remote(batch) for batch in input_batch]
results = [r.result() for r in responses]
print("Result batch is", results)