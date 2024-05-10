from ray import serve
import ray
import pickle
import cv2

@serve.deployment(route_prefix="/image_rotate")
class ImageModel:
    def __init__(self):
        #do any initilization here, like loading a model!
        pass

    async def __call__(self, starlette_request):
        # -- get bytes from the request and convert it to an image
        image_payload_bytes = await starlette_request.body()
        img = pickle.loads(image_payload_bytes)

        # -- process the image normally
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mean_val = img.mean()

        # -- return our results by converting them to a byte stream
        return pickle.dumps((img, mean_val))

runner = ImageModel.bind()
ray.init()
serve.start(http_options={"host": "0.0.0.0", "port":8008})