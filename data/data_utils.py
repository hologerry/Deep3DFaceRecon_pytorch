import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR, cvtColor, imread
from skimage import img_as_float32, img_as_ubyte


def get_params(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    # augmentation_params:
    # flip_param:
    #     horizontal_flip: True
    #     time_flip: True
    # jitter_param:
    #     brightness: 0.1
    #     contrast: 0.1
    #     saturation: 0.1
    #     hue: 0.1
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def crop_square_tensor(img, size=256, interpolation=TF.InterpolationMode.BICUBIC):
    h, w = img.shape[1:]  # C, H, W
    if h == size and w == size:
        return img
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        :, int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2), int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2)
    ]
    resized = TF.resize(crop_img, (size, size), interpolation=interpolation, antialias=True)

    return resized


def crop_square_video_tensor(video, size=256, interpolation=TF.InterpolationMode.BICUBIC):
    # video: T, C, H, W
    T, C, H, W = video.shape
    min_size = np.amin([H, W])

    # Centralize and crop
    crop_video = video[
        :,
        :,
        int(H / 2 - min_size / 2) : int(H / 2 + min_size / 2),
        int(W / 2 - min_size / 2) : int(W / 2 + min_size / 2),
    ]
    resize_images = [
        TF.resize(crop_img, (size, size), interpolation=interpolation, antialias=True) for crop_img in crop_video
    ]
    resized = torch.stack(resize_images, dim=0)

    return resized


def crop_square_ndarray(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    if h >= size:
        resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return resized


def crop_square_ndarray_upside(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        h - min_size : h,
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    if h >= size:
        resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return resized


def crop_square(img, size=256):
    if isinstance(img, np.ndarray):
        return crop_square_ndarray(img, size=size)
    elif isinstance(img, torch.Tensor):
        return crop_square_tensor(img, size=size)


def read_image(image_path, size=256, to_cuda=False):
    img = imread(image_path)
    img = crop_square(img, size=size)
    img = img_as_float32(cvtColor(img, COLOR_BGR2RGB))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if to_cuda:
        img = img.unsqueeze(0).cuda()

    return img


def cvt_image(image, size=256, to_cuda=False):
    img = crop_square(image, size=size)
    img = img_as_float32(cvtColor(img, COLOR_BGR2RGB))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img)
    if to_cuda:
        img = img.unsqueeze(0).cuda()

    return img


def image_to_numpy(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        out_img = np.transpose(image.data.cpu().numpy(), [0, 2, 3, 1])[0]
        out_img = img_as_ubyte(out_img)

        out_img = cvtColor(out_img, COLOR_RGB2BGR)
        return out_img
