import imageio
import torch


def write_gif(frames, output_path):
    with imageio.get_writer(output_path, mode='I', duration=0.03) as writer:
        for frame in frames:
            # Convert the NumPy array to uint8 (expected format by imageio)
            writer.append_data(frame)

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
