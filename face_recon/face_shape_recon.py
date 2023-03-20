import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .bfm import ParametricFaceModel
from .networks import ReconNetWrapper


class FaceShapeRecon(nn.Module):
    def __init__(
        self,
        bfm_folder="./BFM",
        checkpoint_path="./checkpoints/talking_head_all_recon_8gpu_80ep/epoch_20.pth",
        net_recon="resnet50",
        device="cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        focal = 1015.0
        center = 112.0
        camera_d = 10.0
        z_near = 5.0
        z_far = 15.0
        self.rescale_factor = 102.0
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.net_recon = ReconNetWrapper(net_recon=net_recon, use_last_fc=False)
        self.net_recon.load_state_dict(checkpoint["net_recon"])
        self.net_recon.to(device)
        self.net_recon.eval()

        self.face_model = ParametricFaceModel(
            bfm_folder=bfm_folder,
            camera_distance=camera_d,
            focal=focal,
            center=center,
            is_train=False,
            default_name="BFM_model_front.mat",
            device=device,
        )

    def split_coeff(self, coeffs):
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80:144]
        tex_coeffs = coeffs[:, 144:224]
        angles = coeffs[:, 224:227]
        gammas = coeffs[:, 227:254]
        translations = coeffs[:, 254:]
        return {
            "id": id_coeffs,
            "exp": exp_coeffs,
            "tex": tex_coeffs,
            "angle": angles,
            "gamma": gammas,
            "trans": translations,
        }

    def get_coeff(self, input, normalize=False):
        if normalize:
            input = (input + 1.0) / 2.0
        input_resize = TF.resize(input, (224, 224))
        input_coeff = self.net_recon(input_resize)
        preds_dict = self.split_coeff(input_coeff)
        return preds_dict

    def get_landmarks(self, coeff_dict):
        return self.face_model.compute_landmark(coeff_dict)

    def forward(self, input, normalize=False):
        if normalize:
            input = (input + 1.0) / 2.0
        input_resize = TF.resize(input, (224, 224))
        input_coeff = self.net_recon(input_resize)
        preds_dict = self.split_coeff(input_coeff)
        landmark = self.face_model.compute_landmark(preds_dict)
        preds_dict.update({"landmark": landmark})
        return preds_dict
