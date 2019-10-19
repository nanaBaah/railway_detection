import numpy as np
import os
import glob
import cv2 as cv
from matplotlib import pyplot as plt


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b-a) + a


def shrink_images(rgb_image, mask, fraction=0.5):
    img = cv.resize(rgb_image, None, fx=fraction, fy=fraction, interpolation=cv.INTER_AREA)
    mask_img = cv.resize(mask, None, fx=fraction, fy=fraction, interpolation=cv.INTER_AREA)

    return img, mask_img


def enlarge_images(rgb_image, mask, fraction=2.0):
    # cv.INTER_CUBIC for best but slow processing

    img = cv.resize(rgb_image, None, fx=fraction, fy=fraction, interpolation=cv.INTER_LINEAR)
    mask_img = cv.resize(mask, None, fx=fraction, fy=fraction, interpolation=cv.INTER_LINEAR)

    return img, mask_img


def vertical_flip(rgb_image, mask):

    if rand() < 0.5:
        img = cv.flip(rgb_image, 1)
        mask_img = cv.flip(mask, 1)
    else:
        img = rgb_image
        mask_img = mask

    return img, mask_img


def change_hsv(rgb_image, mask, hsv_img=5):

    images = []
    masks_img = []

    for idx in range(hsv_img):
        img_hsv = cv.cvtColor(rgb_image, cv.COLOR_BGR2HSV)

        # extrapolate intensity channels
        H = img_hsv[:, :, 0].astype(np.float32)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        hue = rand(-0.1, 0.1) * 255.0
        H += hue
        np.clip(H, a_min=0, a_max=255, out=H)

        # Modify Sat
        sat = rand(1, 1.5) if rand() < 0.5 else 1 / rand(1, 1.5)
        S *= sat
        np.clip(S, a_min=0, a_max=255, out=S)

        # Modify Val
        val = rand(1, 1.5) if rand() < 0.5 else 1 / rand(1, 1.5)
        V *= val
        np.clip(V, a_min=0, a_max=255, out=V)

        # Concatenate dimensions together again
        img_hsv[:, :, 0] = H.astype(np.uint8)
        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)

        images.append(cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR))
        masks_img.append(mask)

    return np.array(list(zip(images, masks_img)))

