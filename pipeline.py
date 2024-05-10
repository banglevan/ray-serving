from rembg import remove
from matplotlib import pyplot as plt
from PIL import Image
import  random
import cv2
import numpy as np
from pymatting import *
def generate_trimap(alpha):
   k_size = 5#random.choice(range(2, 5))
   iterations = 3#np.random.randint(5, 15)
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
   dilated = cv2.dilate(alpha, kernel, iterations=iterations)
   eroded = cv2.erode(alpha, kernel, iterations=iterations)
   trimap = np.zeros(alpha.shape, dtype=np.uint8)
   trimap.fill(128)

   trimap[eroded >= 255] = 255
   trimap[dilated <= 0] = 0

   return trimap

def advantage_pymat(image, trimap):
   image = image.astype(np.float_) / 255.
   trimap = trimap.astype(np.float_) / 255.

   # estimate alpha from image and trimap
   alpha = estimate_alpha_cf(image, trimap)

   # make gray background
   background = np.zeros(image.shape)
   background[:, :] = [0.5, 0.5, 0.5]

   # estimate foreground from image and alpha
   foreground = estimate_foreground_ml(image, alpha)

   # blend foreground with background and alpha, less color bleeding
   new_image = blend(foreground, background, alpha)

   # save results in a grid
   images = [image, trimap, alpha, new_image]
   grid = make_grid(images)
   save_image("lemur_grid.png", grid)

   # save cutout
   cutout = stack_images(foreground, alpha)
   save_image("lemur_cutout.png", cutout)

   # just blending the image with alpha results in color bleeding
   color_bleeding = blend(image, background, alpha)
   grid = make_grid([color_bleeding, new_image])
   save_image("lemur_color_bleeding.png", grid)
   return alpha

image = Image.open('data/ex1.jpg')
result = remove(data=image, only_mask=True)
result = np.array(result).astype(np.uint8)
trimap = generate_trimap(result)

new_image = advantage_pymat(np.array(image), trimap)

plt.imshow(new_image)
plt.show()
