import os

from argparse import ArgumentParser

import numpy as np
import torch

from torchvision.utils import save_image
from tqdm import tqdm

from data.talkinghead_dataset import get_dataloader
from face_recon.face_shape_recon import FaceShapeRecon


def cat_arrays_dicts(coeffs_dicts):
    coeffs = {}
    for key in coeffs_dicts[0].keys():
        coeffs[key] = np.concatenate([coeffs_dict[key] for coeffs_dict in coeffs_dicts], axis=0)
    return coeffs


@torch.no_grad()
def main(args):

    face_shape_recon = FaceShapeRecon(checkpoint_path="./checkpoints/talking_head_all_recon_8gpu_80ep/epoch_24.pth")
    data_loader = get_dataloader(
        split=args.split, size=256, data_type="two", part_idx=args.part_idx, part_num=args.part_num
    )

    for data_batch in tqdm(data_loader, total=len(data_loader), desc=f"Forwarding {args.part_idx}/{args.part_num}"):
        video = data_batch["video"][0]
        video_name = data_batch["video_name"][0]

        T = video.shape[0]

        cur_video_coeffs_list = []

        os.makedirs(args.output_coeffs_dir, exist_ok=True)
        cur_video_coeffs_output_path = os.path.join(args.output_coeffs_dir, video_name + ".npz")

        for i in range(0, T, args.sub_batch_size):
            sub_video = video[i : i + args.sub_batch_size, ...]
            sub_video = sub_video.cuda()
            sub_coeffs = face_shape_recon(sub_video)
            for k, v in sub_coeffs.items():
                sub_coeffs[k] = v.cpu().numpy()
            cur_video_coeffs_list.append(sub_coeffs)

        cur_video_coeffs = cat_arrays_dicts(cur_video_coeffs_list)
        np.savez(cur_video_coeffs_output_path, **cur_video_coeffs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--sub_batch_size", type=int, default=32)

    parser.add_argument(
        "--split",
        default="train",
        help="path to output landmarks",
    )
    parser.add_argument("--output_coeffs_dir", help="path to output coeffs")

    args = parser.parse_args()

    assert args.job_idx < args.job_num
    assert args.gpu_idx < args.gpu_num
    args.part_num = args.job_num * args.gpu_num
    args.part_idx = args.job_idx * args.gpu_num + args.gpu_idx

    print(
        f"GPU job {args.job_idx}/{args.job_num}, gpu {args.gpu_idx}/{args.gpu_num}, part {args.part_idx}/{args.part_num}"
    )
    args.output_coeffs_dir = f"../data/TalkingHead-1KH_datasets/{args.split}_face_coeffs/"
    os.makedirs(args.output_coeffs_dir, exist_ok=True)

    main(args)
