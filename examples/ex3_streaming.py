import asyncio
from typing import AsyncGenerator
from starlette.requests import Request
from starlette.responses import StreamingResponse
from ray.serve.handle import DeploymentHandle

from ray import serve


@serve.deployment
class StreamingResponder:
    async def generate_numbers(self, max: str) -> AsyncGenerator[str, None]:
        for i in range(max):
            yield str(i)
            await asyncio.sleep(0.1)

    def __call__(self, request: Request) -> StreamingResponse:
        max = int(request.query_params.get("max", "25"))
        gen = self.generate_numbers(max)
        return StreamingResponse(gen, status_code=200, media_type="text/plain")