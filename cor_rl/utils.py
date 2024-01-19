import math
import os


import imageio
import torch

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "rl_results")

MAX_FPS = 50.0


def write_gif(frames, output_path, fps=140):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if fps > MAX_FPS:
        downsample = math.ceil(fps / MAX_FPS)
        new_fps = fps / downsample
        imageio.v3.imwrite(
            output_path, frames[::downsample], duration=int(1000/new_fps), loop=1
        )
    else:
        imageio.v3.imwrite(output_path, frames, duration=int(1000/fps), loop=1)
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
