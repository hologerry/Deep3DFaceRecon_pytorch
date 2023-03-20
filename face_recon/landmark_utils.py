import numpy as np


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


def norm_landmark(landmark):
    landmark[:, :, 0] = landmark[:, :, 0] / 223.0
    landmark[:, :, 1] = 1 - (landmark[:, :, 1] / 223.0)
    landmark = landmark * 2.0 - 1.0
    return landmark


def select_landmarks(landmarks, num_lmk):
    if num_lmk == 68:
        return landmarks
    elif num_lmk == 34:
        return landmarks[:, ::2, :]
    else:
        raise NotImplementedError


def norm_portrait_kp(kp):
    kp = kp / 255.0
    kp = kp * 2.0 - 1.0
    return kp
