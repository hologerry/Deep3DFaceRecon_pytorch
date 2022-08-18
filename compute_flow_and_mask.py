import cv2
import matplotlib
import numpy as np
import torch
import trimesh

from flow_vis import flow_to_image


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision.utils import save_image

from util.nvdiffrast_raw import MeshRenderer


matplotlib.use("Agg")
import numpy as np

from models.bfm import ParametricFaceModel
from models.networks import ReconNetWrapper


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


def read_img(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_square(img, size=size)
    img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).cuda() / 255.0
    return img


def read_img_np(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_square(img, size=size)
    return img


def make_grid(h, w):
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
    xx = xx.view(1, h, w)
    yy = yy.view(1, h, w)
    grid = torch.cat((xx, yy), 0).float()
    return grid


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    grid = make_grid(H, W).to(flo.device)

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode="bilinear", padding_mode="zeros", align_corners=True)

    return output


def visualize_link(img, annotation, output_path, line_type="-*"):
    """
    visualize the linked facial landmarks according to their physical locations
    """
    plt.figure()
    plt.imshow(img)  # show face image
    x = np.array(annotation[:, 0])
    y = 223 - np.array(annotation[:, 1])
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
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


@torch.no_grad()
def main():
    checkpoint = torch.load("checkpoints/talking_head_recon_8gpu_80ep/epoch_19.pth", map_location="cpu")

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

    renderer = MeshRenderer(rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center))

    image_folder = "val_case_video/image"
    test_img_path_1 = "val_case_video/image/000001.png"
    test_img_path_2 = "val_case_video/image/000130.png"
    test_img_path_3 = "val_case_video/image/000501.png"

    image_1 = read_img(test_img_path_1, size=224)
    image_2 = read_img(test_img_path_2, size=224)
    image_3 = read_img(test_img_path_3, size=224)
    image_np_1 = read_img_np(test_img_path_1, size=224)
    image_np_2 = read_img_np(test_img_path_2, size=224)
    image_np_3 = read_img_np(test_img_path_3, size=224)

    coeff_1 = net_recon(image_1)
    coeff_2 = net_recon(image_2)
    x1, x2, y1, y2 = 0, 223, 0, 223
    pre_face_project, pre_face_vertex, _, _, pre_lmk = face_model.compute_for_render(coeff_1)
    cur_face_project, cur_face_vertex, _, _, cur_lmk = face_model.compute_for_render(coeff_2)
    visualize_link(image_np_2, cur_lmk.detach().squeeze().cpu().numpy(), "vis_flow/img2_lmk.png")
    pre_face_project[..., 1] = 223 - pre_face_project[..., 1]
    cur_face_project[..., 1] = 223 - cur_face_project[..., 1]
    print("pre_face_project shape: ", pre_face_project.shape)
    print("cur_face_project shape: ", cur_face_project.shape)

    face_project_pre_to_cur = pre_face_project - cur_face_project
    print("face_project_pre_to_cur shape: ", face_project_pre_to_cur.shape)
    print("face_model.face_buf", face_model.face_buf.shape)
    pred_mask, depth, flow, idv = renderer(cur_face_vertex, face_model.face_buf, feat=face_project_pre_to_cur)

    print("pred_mask shape: ", pred_mask.shape)
    print("depth shape: ", depth.shape)
    print("flow shape: ", flow.shape)
    print("idv shape: ", idv.shape)

    warped_image_1 = warp(image_1, flow)
    save_image(pred_mask, "vis_flow/pred_mask.png")
    save_image(depth, "vis_flow/depth.png")
    save_image(idv, "vis_flow/idv.png")
    save_image(warped_image_1, "vis_flow/warped_image_1.png")
    save_image(image_1, "vis_flow/image_1.png")
    save_image(image_2, "vis_flow/image_2.png")
    flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
    flow_img = flow_to_image(flow_np, convert_to_bgr=True)
    cv2.imwrite("vis_flow/flow_img.png", flow_img)


if __name__ == "__main__":
    focal = 1015.0
    center = 112.0
    camera_d = 10.0
    z_near = 5.0
    z_far = 15.0
    fov = 2 * np.arctan(center / focal) * 180 / np.pi
    main()
