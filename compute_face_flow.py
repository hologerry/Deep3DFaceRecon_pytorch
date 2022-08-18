import os

import numpy as np
import torch

from PIL import Image
from scipy import ndimage

from models import create_model
from options.test_options import TestOptions
from util.load_mats import load_lm3d
from util.preprocess_with_flow import align_img


def get_data_path(root="examples"):

    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith("png") or i.endswith("jpg")]
    lm_path = [i.replace("png", "txt").replace("jpg", "txt") for i in im_path]
    lm_path = [
        os.path.join(i.replace(i.split(os.path.sep)[-1], ""), "detections", i.split(os.path.sep)[-1]) for i in lm_path
    ]

    return im_path, lm_path


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, save_landmark_dir):
    images = []
    landmarks = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, f_names in sorted(os.walk(dir)):
        for fname in f_names:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                l_path = os.path.join(
                    save_landmark_dir,
                    os.path.basename(os.path.dirname(path)),
                    os.path.basename(path).split(".")[0] + ".txt",
                )
                landmarks.append(l_path)
    return sorted(images[: len(images)]), sorted(landmarks[: len(landmarks)])


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    img = Image.open(im_path).convert("RGB")
    W, H = img.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, img, lm, _, s, list_vertex = align_img(img, lm, lm3d_std)
    if to_tensor:
        img = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return img, lm, s, list_vertex, W, H


def main(rank, opt, name="examples"):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    # im_path, lm_path = make_dataset("/media/v-linzhao/Train_VFHQ/", "/media/v-linzhao/Train_VFHQ_five_landmark/")

    # im_path, lm_path = get_data_path(name)
    im_path, lm_path = ["/media/v-linzhao/Test_VFHQ/AounZA8MgFA/00000002.png"], [
        "/media/v-linzhao/Test_VFHQ_five_landmark_oriscale/AounZA8MgFA/00000002.txt"
    ]
    im_path_pre, lm_path_pre = ["/media/v-linzhao/Test_VFHQ/AounZA8MgFA/00000001.png"], [
        "/media/v-linzhao/Test_VFHQ_five_landmark_oriscale/AounZA8MgFA/00000001.txt"
    ]
    lm3d_std = load_lm3d(opt.bfm_folder)

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace(".png", "").replace(".jpg", "")
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor, s, list_vertex, W, H = read_data(im_path[i], lm_path[i], lm3d_std)
        im_tensor_pre, lm_tensor_pre, s_pre, list_vertex_pre, W, H = read_data(
            im_path_pre[i], lm_path_pre[i], lm3d_std
        )
        data = {
            "imgs": im_tensor,
            "lms": lm_tensor,
            "imgs_pre": im_tensor_pre,
            "lms_pre": lm_tensor_pre,
            "s": s,
            "list_vertex": list_vertex,
            "s_pre": s_pre,
            "list_vertex_pre": list_vertex_pre,
            "W": W,
            "H": H,
        }
        model.set_input(data)  # unpack data from data loader
        flow = model.compute_flow().detach().cpu().numpy()  # run inference
        flow = ndimage.zoom(flow / s, (1, 2 / 2, H / int(H * s), W / int(W * s)))
        np.save(img_name + ".npy", flow[i])


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    main(0, opt, opt.img_folder)
