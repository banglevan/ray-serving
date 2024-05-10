from typing import Dict, List
from ray import serve
from ray.serve.handle import DeploymentHandle
import numpy as np
@serve.deployment(
    # These values can be overridden in the Serve config.
    user_config={
        "max_batch_size": 10,
        "batch_wait_timeout_s": 0.5,
    }
)
class Model:
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def __call__(self, multiple_samples: List[int]) -> List[int]:
        # Use numpy's vectorized computation to efficiently process a batch.
        return np.array(multiple_samples) * 2
    #
    def reconfigure(self, user_config: Dict):
        self.__call__.set_max_batch_size(user_config["max_batch_size"])
        self.__call__.set_batch_wait_timeout_s(user_config["batch_wait_timeout_s"])

handle: DeploymentHandle = serve.run(Model.bind()).options(
    use_new_handle_api=True,
)
responses = [handle.remote(i) for i in range(8)]
assert list(r.result() for r in responses) == [i * 2 for i in range(8)]