import cv2
import numpy as np
from PIL import Image

def bytes2image(bimage: bytearray) -> Image:
    npimg = np.frombuffer(bimage, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pimage = Image.fromarray(frame)
    return pimage