import glob
import json
import math
import os
import os.path
import pickle
import random
import struct

from io import BytesIO

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

from imageio import mimread
from PIL import Image
from scipy.io import loadmat, savemat
from skimage import img_as_float32, io
from skimage.color import gray2rgb
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.io import read_video as t_read_video

import util.util as util

from data.base_dataset import (
    BaseDataset,
    apply_img_affine,
    apply_lm_affine,
    get_affine_mat,
    get_transform,
)
from data.data_utils import crop_square_video_tensor, read_image
from data.image_folder import make_dataset
from util.load_mats import load_lm3d
from util.preprocess import align_img, estimate_norm


def crop_square(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


class GMM:
    def __init__(self, dim, num, w, mu, cov, cov_det, cov_inv):
        self.dim = dim  # feature dimension
        self.num = num  # number of Gaussian components
        self.w = w  # weights of Gaussian components (a list of scalars)
        self.mu = mu  # mean of Gaussian components (a list of 1x dim vectors)
        self.cov = cov  # covariance matrix of Gaussian components (a list of dim x dim matrices)
        self.cov_det = cov_det  # pre-computed determined of covariance matrices (a list of scalars)
        self.cov_inv = cov_inv  # pre-computed inverse covariance matrices (a list of dim x dim matrices)

        self.factor = [0] * num
        for i in range(self.num):
            self.factor[i] = (2 * math.pi) ** (self.dim / 2) * self.cov_det[i] ** 0.5

    def likelihood(self, data):
        assert data.shape[1] == self.dim
        N = data.shape[0]
        lh = np.zeros(N)

        for i in range(self.num):
            data_ = data - self.mu[i]

            tmp = np.matmul(data_, self.cov_inv[i]) * data_
            tmp = np.sum(tmp, axis=1)
            power = -0.5 * tmp

            p = np.array([math.exp(power[j]) for j in range(N)])
            p = p / self.factor[i]
            lh += p * self.w[i]

        return lh


def _rgb2ycbcr(rgb):
    m = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]])
    shape = rgb.shape
    rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.0)
    ycbcr[:, 0] += 16.0
    ycbcr[:, 1:] += 128.0
    return ycbcr.reshape(shape)


def _bgr2ycbcr(bgr):
    rgb = bgr[..., ::-1]
    return _rgb2ycbcr(rgb)


gmm_skin_w = [0.24063933, 0.16365987, 0.26034665, 0.33535415]
gmm_skin_mu = [
    np.array([113.71862, 103.39613, 164.08226]),
    np.array([150.19858, 105.18467, 155.51428]),
    np.array([183.92976, 107.62468, 152.71820]),
    np.array([114.90524, 113.59782, 151.38217]),
]
gmm_skin_cov_det = [5692842.5, 5851930.5, 2329131.0, 1585971.0]
gmm_skin_cov_inv = [
    np.array(
        [
            [0.0019472069, 0.0020450759, -0.00060243998],
            [0.0020450759, 0.017700525, 0.0051420014],
            [-0.00060243998, 0.0051420014, 0.0081308950],
        ]
    ),
    np.array(
        [
            [0.0027110141, 0.0011036990, 0.0023122299],
            [0.0011036990, 0.010707724, 0.010742856],
            [0.0023122299, 0.010742856, 0.017481629],
        ]
    ),
    np.array(
        [
            [0.0048026871, 0.00022935172, 0.0077668377],
            [0.00022935172, 0.011729696, 0.0081661865],
            [0.0077668377, 0.0081661865, 0.025374353],
        ]
    ),
    np.array(
        [
            [0.0011989699, 0.0022453172, -0.0010748957],
            [0.0022453172, 0.047758564, 0.020332102],
            [-0.0010748957, 0.020332102, 0.024502251],
        ]
    ),
]

gmm_skin = GMM(3, 4, gmm_skin_w, gmm_skin_mu, [], gmm_skin_cov_det, gmm_skin_cov_inv)

gmm_nonskin_w = [0.12791070, 0.31130761, 0.34245777, 0.21832393]
gmm_nonskin_mu = [
    np.array([99.200851, 112.07533, 140.20602]),
    np.array([110.91392, 125.52969, 130.19237]),
    np.array([129.75864, 129.96107, 126.96808]),
    np.array([112.29587, 128.85121, 129.05431]),
]
gmm_nonskin_cov_det = [458703648.0, 6466488.0, 90611376.0, 133097.63]
gmm_nonskin_cov_inv = [
    np.array(
        [
            [0.00085371657, 0.00071197288, 0.00023958916],
            [0.00071197288, 0.0025935620, 0.00076557708],
            [0.00023958916, 0.00076557708, 0.0015042332],
        ]
    ),
    np.array(
        [
            [0.00024650150, 0.00045542428, 0.00015019422],
            [0.00045542428, 0.026412144, 0.018419769],
            [0.00015019422, 0.018419769, 0.037497383],
        ]
    ),
    np.array(
        [
            [0.00037054974, 0.00038146760, 0.00040408765],
            [0.00038146760, 0.0085505722, 0.0079136286],
            [0.00040408765, 0.0079136286, 0.010982352],
        ]
    ),
    np.array(
        [
            [0.00013709733, 0.00051228428, 0.00012777430],
            [0.00051228428, 0.28237113, 0.10528370],
            [0.00012777430, 0.10528370, 0.23468947],
        ]
    ),
]

gmm_nonskin = GMM(3, 4, gmm_nonskin_w, gmm_nonskin_mu, [], gmm_nonskin_cov_det, gmm_nonskin_cov_inv)

prior_skin = 0.8
prior_nonskin = 1 - prior_skin


def skinmask(imbgr):
    im = _bgr2ycbcr(imbgr)

    data = im.reshape((-1, 3))

    lh_skin = gmm_skin.likelihood(data)
    lh_nonskin = gmm_nonskin.likelihood(data)

    tmp1 = prior_skin * lh_skin
    tmp2 = prior_nonskin * lh_nonskin
    post_skin = tmp1 / (tmp1 + tmp2)  # posterior probability

    post_skin = post_skin.reshape((im.shape[0], im.shape[1]))

    post_skin = np.round(post_skin * 255)
    post_skin = post_skin.astype(np.uint8)
    post_skin = np.tile(np.expand_dims(post_skin, 2), [1, 1, 3])  # reshape to H*W*3

    return post_skin


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


class CelebvhqDataset(BaseDataset):
    """
    Dataset of videos, each video can be represented as:
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(
        self,
        opt,
    ):
        super().__init__(opt)
        self.lm3d_std = load_lm3d(opt.bfm_folder)
        self.opt = opt

        if self.opt.isTrain:
            self.vide_names_json = os.path.join(opt.data_root, opt.celebvhq_video_names_json)
            self.video_dir = os.path.join(opt.data_root, opt.celebvhq_video_dir)
            self.landmark_dir = os.path.join(opt.data_root, opt.celebvhq_landmark_dir)
        else:
            self.vide_names_json = os.path.join(opt.data_root, opt.celebvhq_video_names_json_val)
            self.video_dir = os.path.join(opt.data_root, opt.celebvhq_video_dir_val)
            self.landmark_dir = os.path.join(opt.data_root, opt.celebvhq_landmark_dir_val)

        self.videos = json.load(open(self.vide_names_json))["videos"]
        if self.opt.isTrain:
            self.videos = self.videos[:-500]
        else:
            self.videos = self.videos[-500:]
        self.name = "video_dataset"

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        if ".mp4" not in video_name:
            video_name += ".mp4"
        video_basename = video_name.split(".")[0]
        video_file = os.path.join(self.video_dir, video_name)
        landmark_json_file = os.path.join(self.landmark_dir, video_basename + ".json")
        with open(landmark_json_file) as f:
            landmark_dict = json.load(f)

        # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_idx = random.randint(0, frame_count - 1)
        # while str(frame_idx) not in landmark_dict:
        #     frame_idx = random.randint(0, frame_count - 1)

        frame_ids_all = landmark_dict.keys()
        frame_ids_all = [int(x) for x in frame_ids_all]
        frame_idx = random.choice(frame_ids_all)
        landmark = landmark_dict[str(frame_idx)]

        video = cv2.VideoCapture(video_file)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, frame = video.read()

        try_num = 1

        while try_num < 5 and (not res or frame is None):
            try_num += 1
            frame_idx = random.choice(frame_ids_all)
            landmark = landmark_dict[str(frame_idx)]

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            res, frame = video.read()

        if not res or frame is None:
            pre_idx = idx - 1
            nxt_idx = idx + 1
            if pre_idx >= 0:
                return self.__getitem__(pre_idx)
            else:
                return self.__getitem__(nxt_idx)
        frame = crop_square(frame, 224)
        landmark = np.array(landmark) / 255.0 * 223.0
        landmark[:, 1] = 223 - landmark[:, 1]
        mask = skinmask(frame)

        M = estimate_norm(landmark, 224)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        mask_tensor = mask_tensor[:1, :, :]
        landmark_tensor = torch.from_numpy(landmark).float()
        M_tensor = torch.from_numpy(M).float()

        if self.opt.isTrain:
            brightness, contrast, saturation, hue = get_params()
            img_tensor = TF.adjust_brightness(img_tensor, brightness)
            img_tensor = TF.adjust_contrast(img_tensor, contrast)
            img_tensor = TF.adjust_saturation(img_tensor, saturation)
            img_tensor = TF.adjust_hue(img_tensor, hue)

        return {
            "imgs": img_tensor,
            "lms": landmark_tensor,
            "msks": mask_tensor,
            "M": M_tensor,
        }


class CelebvhqInferDataset(data.Dataset):
    def __init__(self, size=224, part_idx=0, part_num=1):
        self.size = size
        self.part_idx = part_idx
        self.part_num = part_num
        self.vide_names_json = os.path.join("../data", "CelebV-HQ_downloaded", "processed_35666_videos_names.json")
        self.video_dir = os.path.join("../data", "CelebV-HQ_downloaded", "processed_35666")

        with open(self.vide_names_json) as f:
            all_videos = json.load(f)["videos"]

        self.videos = all_videos[self.part_idx :: self.part_num]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        if ".mp4" not in video_name:
            video_name += ".mp4"
        video_basename = video_name.split(".")[0]
        video_file = os.path.join(self.video_dir, video_name)

        video, _, meta = read_video(video_file, pts_unit="sec")

        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)

        # fps = meta["video_fps"]
        video = crop_square_video_tensor(video, size=self.size)

        out = {"video": video, "video_name": video_basename}
        return out


def get_dataloader(split="train", size=224, data_type="two", part_idx=0, part_num=1):
    dataset = CelebvhqInferDataset(size=size, part_idx=part_idx, part_num=part_num)
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=16, drop_last=False)
    return loader
