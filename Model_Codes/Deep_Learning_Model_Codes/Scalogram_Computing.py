import os
import numpy as np
import pandas as pd
import matplotlib
from scipy.signal import detrend, butter, filtfilt
from tqdm import tqdm
from PIL import Image

excel_dir = "/content/drive/MyDrive/PT_Data"
out_dir = "/content/drive/MyDrive/Morse_Scalo_Signals_PT"
classes   = ["preictal", "interictal"]

fs = 250
chunk_seconds = 10
image_size = (224, 224)
colormap = matplotlib.colormaps["jet"]

freq_min = 0.5
freq_max = 40.0
num_freqs = 80

beta = 3.0
gamma = 10.0

os.makedirs(out_dir, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

def bandpass_filter(x, fs, low=0.5, high=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def scale_power_to_rgb(power, freqs, freq_band, vmin=None, vmax=None):
    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    power = power[mask, :]
    power_db = 10 * np.log10(power + 1e-12)
    if vmin is None:
        vmin = np.percentile(power_db, 5)
    if vmax is None:
        vmax = np.percentile(power_db, 95)
    power_db = np.clip(power_db, vmin, vmax)
    normed = (power_db - vmin) / (vmax - vmin + 1e-12)
    rgba = colormap(normed)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

def morse_cwt_fft(x, fs, beta=3.0, gamma=10.0, freqs_hz=None):
    N = len(x)
    Xf = np.fft.fft(x)
    omega = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0/fs)  # rad/s, includes negatives
    omega_abs = np.abs(omega)
    # analytic (positive) mask
    pos_mask = omega > 0
    # peak (angular) frequency of Morse wavelet (dimensionless)
    omega_p = (beta / gamma) ** (1.0 / gamma)
    if freqs_hz is None:
        freqs_hz = np.linspace(freq_min, freq_max, num_freqs)
    scales = omega_p / (2.0 * np.pi * freqs_hz)  # scale -> corresponds to each target freq
    cwtmat = np.zeros((len(scales), N), dtype=np.complex64)
    for i, s in enumerate(scales):
        arg = s * omega_abs
        # Morse frequency-domain (unnormalized) for positive freqs, zero for non-positive to enforce analyticity
        psi_hat = np.zeros_like(omega, dtype=np.complex64)
        # avoid tiny values at zero
        arg_pos = arg.copy()
        # compute only where omega>0
        pos_idx = pos_mask
        psi_hat[pos_idx] = (arg_pos[pos_idx] ** beta) * np.exp(- (arg_pos[pos_idx] ** gamma))
        # multiply by Xf * conj(psi_hat) and inverse fft
        Wf = Xf * np.conjugate(psi_hat)
        w = np.fft.ifft(Wf)
        cwtmat[i, :] = w
    # freqs_hz are our pseudo-frequencies corresponding to rows
    return cwtmat, freqs_hz

for cls in classes:
    files = [f for f in os.listdir(excel_dir) if f.lower().startswith(f"ecg_chunks_{cls}")]
    for fname in files:
        path = os.path.join(excel_dir, fname)
        df = pd.read_excel(path, header=None)
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{cls}:{fname}", leave=False):
            arr = row.dropna().to_numpy(dtype=float)
            expected_len = fs * chunk_seconds
            if arr.size > expected_len:
                arr = arr[:expected_len]
            elif arr.size < expected_len:
                arr = np.pad(arr, (0, expected_len - arr.size))
            x = detrend(arr)
            x = bandpass_filter(x, fs, low=0.5, high=40, order=4)
            x = (x - x.mean()) / (x.std() + 1e-12)
            cwtm, freqs = morse_cwt_fft(x, fs, beta=beta, gamma=gamma,
                                       freqs_hz=np.linspace(freq_min, freq_max, num_freqs))
            power = np.abs(cwtm) ** 2
            rgb = scale_power_to_rgb(power, freqs, (freq_min, freq_max), vmin=-25, vmax=25)
            img = Image.fromarray(rgb)
            img = img.resize(image_size, Image.BICUBIC)
            out_path = os.path.join(out_dir, cls, f"{os.path.splitext(fname)[0]}_row{row_idx}.png")
            img.save(out_path, format="PNG", quality=95)

print("Done: Morse-like scalograms saved to", out_dir)
