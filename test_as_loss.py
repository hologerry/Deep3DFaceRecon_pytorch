import os

import cv2
import dlib
import numpy as np
import torch
import trimesh

from PIL import Image
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

from models.bfm import ParametricFaceModel
from models.networks import ReconNetWrapper
from util.load_mats import load_lm3d
from util.nvdiffrast import MeshRenderer


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


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def lmk_to_np(shape, dtype="int32"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


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
            f.write(str(lmks[i][0]) + " " + str(lmks[i][0]))  # x,y coordiantes

            if i != len(lmks) - 1:
                f.write("\n")


def POS(xp, x):
    print("xp shape: ", xp.shape)
    print("x shape: ", x.shape)
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0 : 2 * npts - 1 : 2, 0:3] = x.transpose()
    A[0 : 2 * npts - 1 : 2, 3] = 1

    A[1 : 2 * npts : 2, 4:7] = x.transpose()
    A[1 : 2 * npts : 2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def resize_n_crop_img(img, lm, t, s, target_size=224.0, mask=None):
    h0, w0 = img.shape[:2]
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size
    left = int(left)
    right = int(right)
    up = int(up)
    below = int(below)

    img = cv2.resize(img, (w, h))
    # img = img.resize((w, h), resample=Image.BICUBIC)
    # img = img.crop((left, up, right, below))
    img = img[up:below, left:right, :]

    # if mask is not None:
    #     mask = mask.resize((w, h), resample=Image.BICUBIC)
    #     mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    return img, lm, mask


def resize_n_crop_img_tensor(img_tensor, lm, t, s, target_size=224.0, mask=None):
    # img_tensor is tensor (C, H, W), others are numpy
    _, h0, w0 = img_tensor.shape
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size
    left = int(left)
    right = int(right)
    up = int(up)
    below = int(below)

    img_new_tensor = resize(img_tensor, (h, w))
    img_new_tensor = img_new_tensor[:, up:below, left:right]

    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    return img_new_tensor, lm, mask


def align_img(img, lm, lm3D, mask=None, target_size=224.0, rescale_factor=102.0):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (5, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (5, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    h0, w0 = img.shape[:2]
    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm.transpose(), lm3D.transpose())
    s = rescale_factor / s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new


def align_img_tensor(img_tensor, lm, lm3D, target_size=224.0, rescale_factor=102.0):
    _, h0, w0 = img_tensor.shape
    print("img_tensor shape", img_tensor.shape)
    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm.transpose(), lm3D.transpose())
    s = rescale_factor / s
    print("t: ", t.shape)
    print("s: ", s.shape)
    print("s", s)

    # processing the image
    img_new_tensor, lm_new, mask_new = resize_n_crop_img_tensor(img_tensor, lm, t, s, target_size=target_size)
    trans_params = np.array([w0, h0, s, t[0], t[1]], dtype=np.float32)

    return trans_params, img_new_tensor, lm_new, mask_new


def save_mesh(face_model, pred_vertex, pred_color, file_name):
    recon_shape = pred_vertex  # get reconstructed shape
    recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
    recon_shape = recon_shape.detach().cpu().numpy()[0]
    recon_color = pred_color
    recon_color = recon_color.detach().cpu().numpy()[0]
    tri = face_model.face_buf.cpu().numpy()
    mesh = trimesh.Trimesh(
        vertices=recon_shape, faces=tri, vertex_colors=np.clip(255.0 * recon_color, 0, 255).astype(np.uint8)
    )
    mesh.export(file_name)


def main(net_recon, face_model, renderer):

    shape_predictor_path = "shape_predictor/shape_predictor_68_face_landmarks.dat"
    # test_img_path = "datasets/examples/000002.jpg"
    test_img_path_1 = "datasets/talkinghead-val/000277_rec.png"
    test_img_path_2 = "datasets/talkinghead-val/000277_rec.png"
    # test_img_path_2 = "datasets/talkinghead-val/000277_rec.png"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    image_1 = cv2.imread(test_img_path_1)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_1 = crop_square(image_1)
    image_2 = cv2.imread(test_img_path_2)
    image_2 = crop_square(image_2)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_tensor_1 = torch.from_numpy(image_1.transpose((2, 0, 1))).float().unsqueeze(0).cuda() / 255.0
    image_tensor_2 = torch.from_numpy(image_2.transpose((2, 0, 1))).float().unsqueeze(0).cuda() / 255.0
    images_tensor = torch.cat([image_tensor_1, image_tensor_2], dim=0)
    images_tensor.requires_grad = True
    lm3d_std = load_lm3d("BFM")

    print("lm3d_std", lm3d_std)

    B, _, H, W = images_tensor.size()

    aligned_images = []
    for b_idx in range(B):
        cur_image_tensor = images_tensor[b_idx]
        image = cur_image_tensor.permute(1, 2, 0)
        image = image.detach().cpu().numpy() * 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"vis_test/numpy_img_{b_idx}.jpg", image)

        rects = detector(image, 1)
        print(f"num of rects {len(rects)}")
        rect = rects[0]
        shape = predictor(image, rect)
        lmks = lmk_to_np(shape)
        lm = extract_5p(lmks)
        # lm_tensor = torch.from_numpy(lm).float().unsqueeze(0)
        print("type shape", type(shape))
        print("type lmks", type(lmks))
        # lm = shape.reshape([-1, 2])
        print(lm)
        lm[:, -1] = H - 1 - lm[:, -1]
        _, cur_im_tensor, _, _ = align_img_tensor(cur_image_tensor, lm, lm3d_std)
        # _, img, lm, _ = align_img(image, lm, lm3d_std)
        # cv2.imwrite("aligned_img_talkinghead.png", img)
        # save_image(img, "aligned_img_tensor.png")

        print("cur_im_tensor shape", cur_im_tensor.shape)
        aligned_images.append(cur_im_tensor)

    aligned_images_tensor = torch.stack(aligned_images, dim=0)
    print("aligned_images_tensor device", aligned_images_tensor.device)
    print("aligned_images_tensor shape", aligned_images_tensor.shape)
    print("aligned_images_tensor grad", aligned_images_tensor.requires_grad)

    output_coeff = net_recon(aligned_images_tensor)
    print("output_coeff grad", output_coeff.requires_grad)
    # print(net_recon)
    print("output_coeff:", output_coeff.shape)
    pred_vertex, pred_tex, pred_color, pred_lm = face_model.compute_for_render(output_coeff)
    print("pred_vertex:", pred_vertex.shape)
    print("pred_vertex grad:", pred_vertex.requires_grad)
    print("pred_tex:", pred_tex.shape)
    print("pred_color:", pred_color.shape)
    print("pred_lm:", pred_lm.shape)
    pred_mask, _, pred_face = renderer(pred_vertex, face_model.face_buf, feat=pred_color)
    pred_coeffs_dict = face_model.split_coeff(output_coeff)
    save_mesh(face_model, pred_vertex, pred_color, "vis_test/00027_rec_pred_mesh.obj")
    save_image(aligned_images_tensor, "vis_test/aligned_img_tensor.png")


if __name__ == "__main__":

    focal = 1015.0
    center = 112.0
    camera_d = 10.0
    z_near = 5.0
    z_far = 15.0

    checkpoint = torch.load("checkpoints/face_recon_feat0.2_augment/epoch_20.pth", map_location="cpu")
    print("checkpoint keys", checkpoint.keys())

    net_recon = ReconNetWrapper(net_recon="resnet50", use_last_fc=False)
    net_recon.load_state_dict(checkpoint["net_recon"])
    net_recon.cuda()
    net_recon.eval()

    face_model = ParametricFaceModel(
        bfm_folder="BFM",
        camera_distance=camera_d,
        focal=focal,
        center=center,
        is_train=False,
        default_name="BFM_model_front.mat",
        device="cuda",
    )
    fov = 2 * np.arctan(center / focal) * 180.0 / np.pi
    renderer = MeshRenderer(rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center)).cuda()
    main(net_recon, face_model, renderer)
