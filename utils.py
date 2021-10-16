"""
Some codes from https://https://github.com/elliottwu/DeepHDR
"""

import cv2
import numpy as np
import h5py
import torch


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)



MU = 5000. # tunemapping parameter
def tonemap_np(images):  # input/output 0~1
    return np.log(1 + MU * images) / np.log(1 + MU)


def tonemap(images):  # input/output 0~1
    return torch.log(1.0 + MU * images) / np.log(1.0 + MU)


GAMMA = 2.2 # LDR&HDR domain transform parameter
def LDR2HDR(img, expo): # input/output 0~1
    return img ** GAMMA / expo


def arrenge(img):
    H, W, C = img.shape
    img_ = np.zeros(shape=(W, H, C))
    for i in range(W):
        for j in range(H):
            img_[i, j, :] = img[j, i, :]
    return img_


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    _, _, h, w = x.shape
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return x[:, :, max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)]


def resize(x, scale_factor):
    _, _, h, w = x.shape
    x = np.transpose(np.squeeze(x), axes=(1, 2, 0))
    x = np.clip(cv2.resize(x, dsize=(w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC), 0, 1)
    x = np.expand_dims(np.transpose(x, axes=(2, 0, 1)), axis=0)
    return x


def get_input(scene_dir, image_size, scale_factor):
    hf = h5py.File(scene_dir)
    inputs = np.array(hf.get('IN'))
    inputs = center_crop(inputs, image_size)
    inputs = resize(inputs, scale_factor)
    ref_HDR = np.array(hf.get('GT'))
    ref_HDR = center_crop(ref_HDR, image_size)
    return inputs, ref_HDR

def get_image2HDR(image_path, exp, image_size=None, is_crop=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(LDR2HDR(imread(image_path), exp), image_size, is_crop)

# always return RGB, float32, range -1~1
def get_image(image_path, image_size=None, is_crop=False, is_SR=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(imread(image_path), image_size, is_crop, is_SR)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imread(path):
    if path[-4:] == '.hdr':
        img = cv2.imread(path, -1)
    else:
        img = cv2.imread(path)/255.
    return img.astype(np.float32)[...,::-1]

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, path):
    if (path[-4:] == '.hdr'):
        #pdb.set_trace()
        return radiance_writer(path, images)
    else:
        return cv2.imwrite(path, images[...,::-1]*255.)


def transform(image, image_size, is_crop, is_SR=False):
    # npx : # of pixels width/height of image
    if is_crop:
        out = center_crop(image, image_size)
    elif (image_size is not None):
        out = cv2.resize(image, image_size)
    else:
        out = image
    if is_SR:
        out = cv2.resize(out, (image_size[1] // 2, image_size[0] // 2))
    return out.astype(np.float32)

def inverse_transform(images):
    return (images+1.)/2.
