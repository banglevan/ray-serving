import os

import matplotlib.pyplot as plt
import torch.optim
import numpy as np
import cv2
# from mmedit.core.evaluation.metrics import mse, sad, gradient_error, connectivity
from matting_network import MobileMatting
# from pipeline import generate_trimap
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# p1 = '/home/kzx/Compositon1K/test/trimaps'
# p2 = '/home/kzx/Compositon1K/test/merged'
# p3 = '/home/kzx/Compositon1K/test/alpha'
# p4 = './p'
p5 = './lite_matting.ckpt'
patch_hw = 640
def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def trimap_transform(trimap):
    h, w = trimap.shape[0], trimap.shape[1]
    clicks = np.zeros((h, w, 2))
    for k in range(2):
        if (np.count_nonzero(trimap[:, :, k]) > 0):
            dt_mask = -dt(1 - trimap[:, :, k]) ** 2
            L = 320
            clicks[:, :, k] = np.exp(dt_mask / (2 * ((0.05 * L) ** 2)))
    return clicks


def genwmap(qsss, www):
    h, w = qsss + www, qsss + www
    www1 = np.ones([h, w], np.float32) * www
    ws = np.zeros([h, w], np.float32)
    for xx in range(h):
        for yy in range(w):
            ws[xx, yy] = min(xx, yy, h - xx - 1, w - yy - 1) + 1
    ws = ws / (www1 + 1)
    return ws


def run_on_image(rawimg, trimap):
    trimap_nonp = trimap.copy()
    h, w, c = rawimg.shape
    nonph, nonpw, _ = rawimg.shape
    # padh = 0
    # padw = 0
    padh1, padw1 = 0, 0
    padh2, padw2 = 0, 0
    if h < patch_hw:
        padh = patch_hw - h
        padh1 = int(padh / 2)
        padh2 = padh - padh1
    if w < patch_hw:
        padw = patch_hw - w
        padw1 = int(padw / 2)
        padw2 = padw - padw1

    rawimg_pad = cv2.copyMakeBorder(rawimg, padh1, padh2, padw1, padw2, cv2.BORDER_REPLICATE)
    trimap_pad = cv2.copyMakeBorder(trimap, padh1, padh2, padw1, padw2, cv2.BORDER_CONSTANT, value=0)
    h_pad, w_pad, _ = rawimg_pad.shape

    raw_h, raw_w = h_pad, w_pad

    l_step = 320  # qsss
    windows_remain = 320  # psss
    window_siez = windows_remain + l_step
    # window_siez_half = window_siez // 2

    wx = (raw_w - 1 - windows_remain) // l_step + 1  # 640-1-320//320+1
    hx = (raw_h - 1 - windows_remain) // l_step + 1

    # signal_w = wx - 1
    # signal_h = hx - 1
    #
    # alphalist = []
    signal_ws = raw_w - window_siez
    signal_hs = raw_h - window_siez

    img_h = rawimg_pad.copy()
    trimap_h = trimap_pad.copy()

    # alls = []
    # alltri = []
    #
    tp = np.zeros((raw_h, raw_w, 16), np.float32)
    wp = np.zeros((raw_h, raw_w, 16), np.float32)
    apPP = np.zeros((raw_h, raw_w, 16), np.float32)
    # allp = set(list(np.arange(0, 16)))
    windows_sum_weight = genwmap(l_step, windows_remain)

    trimap_tric = np.zeros([*trimap_h.shape, 3], np.float32)
    trimap_tric[:, :, 0] = (trimap_h == 0)
    trimap_tric[:, :, 1] = (trimap_h == 128)
    trimap_tric[:, :, 2] = (trimap_h == 255)

    tritemp = np.zeros([*trimap_h.shape, 2], np.float32)
    tritemp[:, :, 0] = (trimap_h == 0)
    tritemp[:, :, 1] = (trimap_h == 255)
    sixc = trimap_transform(tritemp)

    for h_idx in range(hx):
        for w_idx in range(wx):
            if w_idx == (wx - 1):
                cropw1 = signal_ws
                cropw2 = signal_ws + window_siez

            else:
                cropw1 = w_idx * l_step
                cropw2 = w_idx * l_step + window_siez

            if h_idx == (hx - 1):
                croph1 = signal_hs
                croph2 = signal_hs + window_siez
            else:
                croph1 = h_idx * l_step
                croph2 = h_idx * l_step + window_siez

            tripatch = trimap_tric[croph1:croph2, cropw1:cropw2]
            sixcpatch = sixc[croph1:croph2, cropw1:cropw2]
            if np.sum(tripatch[:, :, 1]) > 1:

                nidx = 0
                for zzz in range(16):
                    if np.sum(tp[croph1:croph2, cropw1:cropw2, zzz]) == 0:
                        nidx = zzz
                        break
                wp[croph1:croph2, cropw1:cropw2, nidx] = windows_sum_weight
                tp[croph1:croph2, cropw1:cropw2, nidx] = 1

                imgpatch = img_h[croph1:croph2, cropw1:cropw2, :]

                with torch.no_grad():
                    timgpatch = torch.from_numpy(
                        imgpatch[:, :, ::-1].transpose((2, 0, 1))[None, :, :, :].astype(np.float32) / 255.).cuda()
                    ttripatch = torch.from_numpy(
                        tripatch.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)).cuda()
                    sixcpatch = torch.from_numpy(
                        sixcpatch.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)).cuda()
                    predFBA1 = segmodel(timgpatch, ttripatch, sixcpatch)

                preda = predFBA1.detach().cpu()
                ap = preda[0, 0].numpy() * tripatch[:, :, 1] + tripatch[:, :, 2]
                apPP[croph1:croph2, cropw1:cropw2, nidx] = ap

    palpha = np.sum(apPP * wp, 2) / np.sum(wp, 2)
    palpha = np.clip(palpha * 255., 0, 255)
    palpha = np.array(palpha, np.uint8)
    wholealpha = palpha[padh1:padh1 + nonph, padw1:padw1 + nonpw]
    wholealpha[trimap_nonp == 0] = 0
    wholealpha[trimap_nonp == 255] = 255
    return wholealpha
def generate_trimap(alpha):
   k_size = 15#random.choice(range(2, 5))
   iterations = 5#np.random.randint(5, 15)
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
   eroded = cv2.erode(alpha.copy(), kernel, iterations=iterations)
   dilated = cv2.dilate(alpha, kernel, iterations=iterations)
   trimap = np.zeros(alpha.shape, dtype=np.uint8)
   trimap.fill(128)

   trimap[eroded >= 10] = 255
   trimap[dilated == 0] = 0

   return trimap
if __name__ == '__main__':
    segmodel = MobileMatting()
    segmodel.load_state_dict(torch.load(p5, map_location='cpu'), strict=False)
    segmodel = segmodel.cuda()
    segmodel.eval()
    from PIL import Image
    from rembg import remove
    image = Image.open('data/ex1.jpg')
    result = remove(data=image, only_mask=True)
    rawimage = np.array(image)
    rawimage = cv2.cvtColor(rawimage, cv2.COLOR_RGB2BGR)

    result = np.array(result).astype(np.uint8)

    blurred = cv2.GaussianBlur(result, (7, 7), 0)
    _,mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    trimap = generate_trimap(result)
    alpha = run_on_image(rawimage, trimap)
    visual = cv2.cvtColor(rawimage.copy(), cv2.COLOR_BGR2RGBA)
    alpha[alpha > 50] = 255
    alpha[alpha <= 50] = 0
    visual[:, :,3] = alpha
    plt.imshow(visual)
    plt.show()
