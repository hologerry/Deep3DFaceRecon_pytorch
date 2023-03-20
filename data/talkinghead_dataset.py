import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

from data.data_utils import read_image


class VFHQDataset(data.Dataset):
    def __init__(self, split, size=256, data_type="two", part_idx=0, part_num=1) -> None:
        super().__init__()
        self.size = size
        self.split = split
        self.part_idx = part_idx
        self.part_num = part_num
        self.data_type = data_type

        if split == "train":
            self.video_dir = "../data/TalkingHead-1KH_datasets/train/cropped_clips_images_256"
            self.files_names_json = "../data/TalkingHead-1KH_datasets/train/cropped_clips_videos_images.json"
            self.coeffs_dir = "../data/TalkingHead-1KH_datasets/train/cropped_clips_coeffs"
        else:
            self.video_dir = "../data/TalkingHead-1KH_datasets/val/cropped_clips_images"
            self.files_names_json = "../data/TalkingHead-1KH_datasets/val/cropped_clips_videos_images.json"
            self.coeffs_dir = "../data/TalkingHead-1KH_datasets/val/cropped_clips_coeffs"

        with open(self.files_names_json, "r") as f:
            self.data_dict = json.load(f)

        all_videos = self.data_dict["videos"]
        cur_part_videos = all_videos[self.part_idx :: self.part_num]
        self.videos = cur_part_videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_name = self.videos[index]
        frame_names = self.data_dict[video_name]

        T = len(frame_names)

        frame_paths = [os.path.join(self.video_dir, video_name, frame_name) for frame_name in frame_names]
        frame_paths = [frame_path.strip() for frame_path in frame_paths]
        frames = [read_image(frame_path, self.size) for frame_path in frame_paths]
        video = torch.stack(frames, dim=0)

        if "norm" in self.data_type:
            video = video * 2.0 - 1.0

        out = {"video": video, "video_name": video_name}
        return out


def get_dataloader(split="train", size=256, data_type="two", part_idx=0, part_num=1):
    dataset = VFHQDataset(split=split, size=size, data_type=data_type, part_idx=part_idx, part_num=part_num)
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=16, drop_last=False)
    return loader
