import warnings
import argparse
import os
import sys
import cv2
import numpy as np
import  random
import paddle
import paddleseg
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import predict
from ppmatting.utils import get_image_list, Config, MatBuilder

warnings.simplefilter('ignore')

def generate_trimap(alpha):
   k_size = random.choice(range(2, 5))
   iterations = np.random.randint(5, 15)
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
   dilated = cv2.dilate(alpha, kernel, iterations=iterations)
   eroded = cv2.erode(alpha, kernel, iterations=iterations)
   trimap = np.zeros(alpha.shape, dtype=np.uint8)
   trimap.fill(128)

   trimap[eroded >= 255] = 255
   trimap[dilated <= 0] = 0

   return trimap

def advantage_pymat():
   from pymatting import *
   import numpy as np

   scale = 1.0

   image = load_image("../data/lemur/lemur.png", "RGB", scale, "box")
   trimap = load_image("../data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")

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

def expert_pymat():
   from pymatting import *
   import numpy as np
   import scipy.sparse

   scale = 1.0

   image = load_image("../data/lemur/lemur.png", "RGB", scale, "box")
   trimap = load_image("../data/lemur/lemur_trimap.png", "GRAY", scale, "nearest")

   # height and width of trimap
   h, w = trimap.shape[:2]

   # calculate laplacian matrix
   L = cf_laplacian(image)

   # decompose trimap
   is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

   # constraint weight
   lambda_value = 100.0

   # build constraint pixel selection matrix
   c = lambda_value * is_known
   C = scipy.sparse.diags(c)

   # build constraint value vector
   b = lambda_value * is_fg

   # build linear system
   A = L + C

   # build ichol preconditioner for faster convergence
   A = A.tocsr()
   A.sum_duplicates()
   M = ichol(A)

   # solve linear system with conjugate gradient descent
   x = cg(A, b, M=M)

   # clip and reshape result vector
   alpha = np.clip(x, 0.0, 1.0).reshape(h, w)

   save_image("lemur_alpha.png", alpha)