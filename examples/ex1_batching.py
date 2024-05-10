from typing import List

import numpy as np

from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Model:
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=10)
    async def __call__(self, multiple_samples: List[int]) -> List[int]:
        # Use numpy's vectorized computation to efficiently process a batch.
        return np.array(multiple_samples) * 2

handle: DeploymentHandle = serve.run(Model.bind()).options(
    use_new_handle_api=True,
)
responses = [handle.remote(i) for i in range(8)]
assert list(r.result() for r in responses) == [i * 2 for i in range(8)]