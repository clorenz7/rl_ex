import os

import imageio
import torch

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "rl_results")


def write_gif(frames, output_path, duration=0.03, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"GIF created and saved at {output_path}")


def get_device(use_gpu=False):
    device = torch.device("cpu")
    if use_gpu:
        if torch.backends.mps.is_available():
            print("Accelerating with MPS!")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print("Accelerating with CUDA!")
            device = torch.device("cuda")

    return device
