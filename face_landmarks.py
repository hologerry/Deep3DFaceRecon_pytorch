import argparse
import os

import cv2
import face_alignment
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cv2 import imread
from skimage import io
from tqdm import tqdm


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


def read_image(image_path):
    img = imread(image_path)
    img = crop_square(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# lmk-68p to lmk-5p
def extract_5p(lm, dtype="int32"):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack(
        [
            lm[lm_idx[0], :],
            np.mean(lm[lm_idx[[1, 2]], :], 0),
            np.mean(lm[lm_idx[[3, 4]], :], 0),
            lm[lm_idx[5], :],
            lm[lm_idx[6], :],
        ],
        axis=0,
    )
    lm5p = lm5p[[1, 2, 0, 3, 4], :].astype(dtype)

    return lm5p  # [left_eye, right_eye, nose, left_mouth, right_mouth]


def save_lmks_to_file(lmks, file):
    with open(file, "w") as f:
        for i in range(len(lmks)):
            f.write(str(lmks[i][0]) + " " + str(lmks[i][1]))  # x,y coordinates

            if i != len(lmks) - 1:
                f.write("\n")


def main(folder_list, mode):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    for img_folder in tqdm(folder_list, leave=False):
        names = [i for i in sorted(os.listdir(img_folder)) if "jpg" in i or "png" in i or "jpeg" in i or "PNG" in i]
        base_dir = os.path.dirname(img_folder)
        txt_save_path = os.path.join(base_dir, "face_landmarks")
        os.makedirs(txt_save_path, exist_ok=True)
        vis_save_path = os.path.join(base_dir, "vis_lm68")
        os.makedirs(vis_save_path, exist_ok=True)
        for name in tqdm(names, desc=os.path.basename(img_folder)):
            full_image_name = os.path.join(img_folder, name)
            txt_name_5 = ".".join(name.split(".")[:-1]) + "_5.txt"
            txt_name_68 = ".".join(name.split(".")[:-1]) + "_68.txt"
            full_txt_name_5 = os.path.join(txt_save_path, txt_name_5)  # 5 facial landmark path for each image
            full_txt_name_68 = os.path.join(txt_save_path, txt_name_68)  # 5 facial landmark path for each image
            full_vis_name = os.path.join(vis_save_path, name)
            img = read_image(full_image_name)
            preds, _, bboxs = fa.get_landmarks_from_image(img, return_bboxes=True)
            assert len(preds) == len(bboxs) == 1
            pred = preds[0]
            visualize_link(img, pred, full_vis_name)
            save_lmks_to_file(pred, full_txt_name_68)
            lmks = extract_5p(pred)
            save_lmks_to_file(lmks, full_txt_name_5)


def visualize_link(img, annotation, output_path, line_type="-*"):
    """
    visualize the linked facial landmarks according to their physical locations
    """
    plt.figure()
    plt.imshow(img)  # show face image
    x = np.array(annotation[:, 0])
    y = np.array(annotation[:, 1])
    star = line_type  # plot style, such as '-*'

    plt.plot(x[0:17], y[0:17], star)  # face contour
    plt.plot(x[17:22], y[17:22], star)  # left eyebrow
    plt.plot(x[22:27], y[22:27], star)  # right eyebrow
    plt.plot(x[27:31], y[27:31], star)  # nose
    plt.plot(x[31:36], y[31:36], star)  # nose
    plt.plot(np.hstack([x[36:42], x[36]]), np.hstack([y[36:42], y[36]]), star)  # left eye
    plt.plot(np.hstack([x[42:48], x[42]]), np.hstack([y[42:48], y[42]]), star)  # right eye
    plt.plot(np.hstack([x[48:60], x[48]]), np.hstack([y[48:60], y[48]]), star)  # mouth
    plt.plot(np.hstack([x[60:68], x[60]]), np.hstack([y[60:68], y[60]]), star)  # mouth
    plt.axis("off")
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets", help="root directory for training data")
    parser.add_argument("--img_folder", nargs="+", required=True, help="folders of training images")
    parser.add_argument("--mode", type=str, default="train", help="train or val")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    shape_predictor_path = "shape_predictor/shape_predictor_5_face_landmarks.dat"
    print("Datasets:", opt.img_folder)
    main([os.path.join(opt.data_root, folder) for folder in opt.img_folder], opt.mode)
