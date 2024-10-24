import sys
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from PGD_L1 import GPM
from PGP_Lp import PGD_Lp
from PGD_L0 import PGD_L0
from fw_lib import FW_LP
from dc_lib import LCPP
from numpy import linalg as LA
import cv2
import pywt
import os
import warnings
from skimage.metrics import peak_signal_noise_ratio

# Import the baseline file
sys.path.append("..")
warnings.filterwarnings("ignore")


def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


image_path = f"./Set12/{1:02d}.png"
output_image_repo = f"./result"
wl = "haar"  # wavelet transform type
level = 4  # wavelet transform level
eps = 0.04  # wavelet transform is not completely sparse, ignore value less than eps

p = 0.9  # lp
m = 200
algorithm = "FW"

if not os.path.isdir(output_image_repo):
    os.makedirs(output_image_repo)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) * 1.0 / 255.0
coeffs = pywt.wavedec2(image, wavelet=wl, level=level)
signal_raw, coeff_slices = pywt.coeffs_to_array(coeffs)
signal = np.zeros_like(signal_raw)
signal[np.abs(signal_raw) > eps] = signal_raw[np.abs(signal_raw) > eps]


# A simple way to parallel
def test(x, m, p, i, method):
    x = signal[:, i]
    A = np.random.normal(0.0, 1.0, [m, x.shape[0]])
    y = A @ x
    radius = LA.norm(x, p) ** p
    if radius < 1e-8:  # If all 0, just return
        return np.zeros_like(x)
    x0 = np.zeros_like(x)
    Lf = np.max(LA.eigvals(A.T @ A).real)
    if method == "L1":
        x_star, _, _ = GPM(
            x0,
            1 / Lf,
            LA.norm(x, 1),
            loss=lambda x: 0.5 * LA.norm(A @ x - y) ** 2,
            gradf=lambda x: A.T @ (A @ x - y),
        )
    elif method == "Lp":
        x_star, _, _ = PGD_Lp(
            x0,
            1 / Lf,
            p,
            radius,
            loss=lambda x: 0.5 * LA.norm(A @ x - y) ** 2,
            gradf=lambda x: A.T @ (A @ x - y),
            verbose=False,
        )
    elif method == "L0":
        x_star, _, _ = PGD_L0(
            x0,
            1 / Lf,
            LA.norm(x, 0),
            loss=lambda x: 0.5 * LA.norm(A @ x - y) ** 2,
            gradf=lambda x: A.T @ (A @ x - y),
        )
    elif method == "FW":
        solver = FW_LP(
            x0,
            p,
            radius,
            obj=lambda x: 0.5 * LA.norm(A @ x - y) ** 2,
            grad=lambda x: A.T @ (A @ x - y),
            Lf=Lf,
        )
        x_star, t, _ = solver.solve(mu=1 / Lf, verbose=False)
    elif method == "LCPP":
        epsilon = 0.8 * ((radius - LA.norm(x0, p) ** p) / x0.shape[0]) ** (1 / p)
        theta = 1 / p
        lambda_ = epsilon ** (1 / theta - 1) / theta
        x_star, _, _ = LCPP(
            lambda x: 0.5 * LA.norm(A @ x - y) ** 2,
            lambda x: A.T @ (A @ x - y),
            x0,
            radius,
            lambda_,
            epsilon,
            theta,
            verbose=False,
            Lf=Lf,
            truncate_threshold=0.1,
        )
    return x_star


# modify n_jobs to add more threads
ans = Parallel(n_jobs=32)(delayed(test)(signal, m, p, i, algorithm) for i in range(256))

signal_recovered = np.zeros_like(signal)
for i in range(len(ans)):
    signal_recovered[:, i] = ans[i]
coeffs_from_arr = pywt.array_to_coeffs(
    signal_recovered, coeff_slices, output_format="wavedec2"
)
image_recovery = pywt.waverec2(coeffs_from_arr, wavelet=wl)
cv2.imwrite(output_image_repo + f"/image.jpg", image_recovery * 255.0)
