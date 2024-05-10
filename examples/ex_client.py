# import requests
#
# prompt = "Tell me a story about dogs."
#
# response = requests.post(f"http://localhost:8000/?prompt={prompt}", stream=True)
# response.raise_for_status()
# for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
#     print(chunk, end="")
#
#     # Dogs are the best.
import numpy as np
# from websockets.sync.client import connect
#
# with connect("ws://localhost:8000") as websocket:
#     websocket.send("Space the final")
#     while True:
#         received = websocket.recv()
#         if received == "<<Response Finished>>":
#             break
#         print(received, end="")
#     print("\n")
#
#     websocket.send(" These are the voyages")
#     while True:
#         received = websocket.recv()
#         if received == "<<Response Finished>>":
#             break
#         print(received, end="")
#     print("\n")
# import requests
# import cv2
# import pickle
# from PIL import Image
# import io
# for i in range(5):
#     r = requests.post("http://localhost:9982/test", json={"data": 5})
#     print(r.json())
# path = r"C:\BANGLV\ray_serving\data\output.png"
# file = {'file': open(path, 'rb')}
# resp=requests.post(url="http://127.0.0.1:8008/image_rotate", files=file)
# image = Image.open(io.BytesIO(resp.content))
# import matplotlib.pyplot as plt
# plt.imshow(np.array(image))
# plt.show()
# # print(resp.content)

#
import ray
import requests
import numpy as np

@ray.remote
def send_query(text):
    resp = requests.post(url="http://127.0.0.1:8008/image_rotate", data=text)
    return resp.text

# Let's use Ray to send all queries in parallel
texts = [
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
path = r"C:\BANGLV\ray_serving\data\output.png"
resp=requests.post(url="http://127.0.0.1:9982/image_rotate",
                   data={"data1": '1', "data3": '2', "data2": '3'},
                   files={'file1': open(path, 'rb'),
                          'file2': open(path, 'rb')})
with open(r"C:\BANGLV\ray_serving\data\outputx.png", 'wb') as f:
    f.write(resp.content)