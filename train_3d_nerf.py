import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from nerf_dataset import NerfDataset
from nerf_model import NeRF
from hash_grid_encoder import MultiResHashGrid

################################ IMPORTANT: This model is quite slow, you do not need to run it until it converges.  ###################################

# Position Encoding
class PositionalEncoder(nn.Module):
    """
    Implement the Position Encoding function.
    Defines a function that embeds x to (sin(2^k*pi*x), cos(2^k*pi*x), ...)
    Please note that the input tensor x should be normalized to the range [-1, 1].

    Args:
    x (torch.Tensor): The input tensor to be embedded.
    L (int): The number of levels to embed.

    Returns:
    torch.Tensor: The embedded tensor.
    """
    def __init__(self, data_range, L):
        super(PositionalEncoder, self).__init__()
        self.L = L
        self.data_range = data_range

    def forward(self, x):
        # Get the original shape and device
        orig_shape = list(x.shape)
        device = x.device

        # Scale input from [-1,1] to [-pi,pi]
        x = x * torch.pi

        # Prepare frequencies for 2^[0...L-1]
        freqs = 2.0 ** torch.arange(self.L, device=device)

        # Reshape x for broadcasting
        x_expanded = x.view(*orig_shape, 1)

        # Compute sin and cos embeddings
        x_freqs = x_expanded * freqs.view(*([1] * len(orig_shape)), -1)
        sin_embed = torch.sin(x_freqs)
        cos_embed = torch.cos(x_freqs)

        # Concatenate sin and cos embeddings
        embed = torch.cat([sin_embed, cos_embed], dim=-1)

        # Flatten the last dimension
        embed = embed.view(*orig_shape[:-1], -1)

        # print("embed.shape",embed.shape)

        return embed




def sample_rays(H, W, f, c2w):
    """
    Samples rays from a camera with given height H, width W, focal length f, and camera-to-world matrix c2w.

    Args:
    H (int): The height of the image.
    W (int): The width of the image.
    f (float): The focal length of the camera.
    c2w (torch.Tensor): The 4x4 camera-to-world transformation matrix.

    Returns:
    rays_o (torch.Tensor): The origin of each ray, with shape (W, H, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (W, H, 3).
    """
    # Create a grid of pixel coordinates
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W))
    i = i.cuda()
    j = j.cuda()

    # Convert pixel coordinates to camera coordinates
    dirs = torch.stack([(i - W * 0.5) / f, (j - H * 0.5) / f, torch.ones_like(i)], dim=-1)  # (W, H, 3)

    # Transform camera directions to world directions using the camera-to-world matrix
    rays_d = (dirs[..., None, :] @ c2w[:3, :3].T).squeeze(-1)  # (W, H, 3)
    rays_d = rays_d.squeeze(-2)
    # The origin of each ray is the camera position
    rays_o = c2w[:3, 3].expand(H, W, -1)  # (W, H, 3)

    # print("rays_o.shape",rays_o.shape)
    # print("rays_d.shape",rays_d.shape)

    # import pdb; pdb.set_trace()

    return rays_o, rays_d

    # pass

def sample_points_along_the_ray(tn, tf, N_samples):
    """
    Samples points uniformly along a ray from time t_n to time t_f.

    Args:
    tn (torch.Tensor): The starting point of the ray.
    tf (torch.Tensor): The ending point of the ray.
    N_samples (int): The number of samples to take along the ray.

    Returns:
    torch.Tensor: A tensor of shape (N_samples, ...) containing the sampled points along the ray,
                  where ... corresponds to the shape of tn or tf.
    """
    # Generate random numbers between 0 and 1 for each sample
    uniform_samples = torch.linspace(0.0, 1.0, N_samples, device=tn.device)  # Shape: (N_samples,)

    # Generate uniform samples in the interval [0, 1] for each segment
    random_noise = torch.rand(N_samples, *tn.shape).cuda()  # Shape: (N_samples, ...)
    # Add the random jitter to the uniform samples
    jittered_samples = uniform_samples + (random_noise - 0.5) / N_samples
    jittered_samples = torch.clamp(jittered_samples, 0.0, 1.0)  # Ensure within [0, 1]

    # Calculate the points along the ray by interpolating between tn and tf
    # using the sampled values
    sampled_points = tn[..., None] + jittered_samples * (tf[..., None] - tn[..., None])  # Shape: (N_samples, ...)


    print("sampled_points.shape",sampled_points.shape)
    return sampled_points


def volumn_render(NeRF, rays_o, rays_d, N_samples):
    """
    Performs volume rendering to generate an image from rays.

    Args:
    NeRF (nn.Module): The neural radiance field model.
    rays_o (torch.Tensor): The origin of each ray, with shape (N_rays, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (N_rays, 3).
    N_samples (int): The number of samples to take along each ray.

    Returns:
    torch.Tensor: The rendered RGB image.
    """
        # Define near and far planes
    # Define near and far planes
    tn, tf = 2.0, 6.0  # Example near and far planes

   # Sample points along the rays
    t_vals = torch.linspace(0.0, 1.0, N_samples, device=rays_o.device)  # (N_samples,)
    z_vals = tn + t_vals * (tf - tn)  # (N_samples,)
    z_vals = z_vals.expand(rays_o.shape[0], N_samples)  # (N_rays, N_samples)

    # Compute 3D coordinates of sampled points along rays
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (N_rays, N_samples, 3)

    # Get RGB and density from NeRF
    dirs = rays_d[..., None, :].expand_as(pts)  # (N_rays, N_samples, 3)
    rgb, sigma = NeRF(pts, dirs)  # rgb.shape => (N_rays, N_samples, 3), sigma.shape => (N_rays, N_samples)
    sigma = sigma.squeeze(-1)

    # Compute distances between adjacent points (delta)
    delta = z_vals[..., 1:] - z_vals[..., :-1]  # (N_rays, N_samples - 1)
    delta = torch.cat([delta, torch.tensor([1e10], device=delta.device).expand(rays_o.shape[0], 1)], dim=-1)  # (N_rays, N_samples)

    # Debug shapes
    # print(f"delta.shape: {delta.shape}")  # 应为 (N_rays, N_samples)
    # print(f"sigma.shape: {sigma.shape}")  # 应为 (N_rays, N_samples)

    # Compute alpha (opacity)
    alpha = 1.0 - torch.exp(-delta * torch.relu(sigma))  # (N_rays, N_samples)

    # Compute transmittance
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)  # (N_rays, N_samples)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)  # Shift for correct computation

    # Compute weights
    weights = alpha * T  # (N_rays, N_samples)

    # Compute the final accumulated color
    rgb_image = torch.sum(weights[..., None] * rgb, dim=-2)  # (N_rays, 3)

    return rgb_image


def random_select_rays(H, W, rays_o, rays_d, img, N_rand):
    """
    Randomly select N_rand rays to reduce memory usage.

    Parameters:
    - H: int, height of the image.
    - W: int, width of the image.
    - rays_o: torch.Tensor, original ray origins with shape (H * W, 3).
    - rays_d: torch.Tensor, ray directions with shape (H * W, 3).
    - img: torch.Tensor, image with shape (H * W, 3).
    - N_rand: int, number of random rays to select.

    Returns:
    - selected_rays_o: torch.Tensor, selected ray origins with shape (N_rand, 3).
    - selected_rays_d: torch.Tensor, selected ray directions with shape (N_rand, 3).
    - selected_img: torch.Tensor, selected image pixels with shape (N_rand, 3).
    """
    # Generate coordinates for all pixels in the image
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

    # Randomly select N_rand indices without replacement
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)

    # Select the corresponding coordinates, rays, and image pixels
    select_coords = coords[select_inds].long().to("cpu")  # (N_rand, 2)
    # print("select_coords.shape",select_coords.shape)
    selected_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = torch.tensor(selected_img, dtype=torch.float32)  # Ensure float32 dtype

    # print("selected_rays_o.shape",selected_rays_o.shape)
    # print("selected_rays_d.shape",selected_rays_d.shape)
    # print("selected_img.shape",selected_img.shape)

    # import pdb; pdb.set_trace()

    return selected_rays_o, selected_rays_d, selected_img


def fit_images_and_calculate_psnr(data_path, epochs=2000, learning_rate=5e-4):
    # get available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    dataset = NerfDataset(data_path)

    # create model
    # xyz_encoder = MultiResHashGridEncoder(data_range=[-4, 4], n_levels=16, n_features_per_level=2).to(device)
    # dir_encoder = MultiResHashGridEncoder(data_range=[-1, 1], n_levels=8, n_features_per_level=2).to(device)
    # nerf = NeRF(
    #     xyz_encoder=xyz_encoder,
    #     dir_encoder=dir_encoder,
    #     input_dim=32,
    #     view_dim=16,
    # ).to(device)
    xyz_encoder = PositionalEncoder(data_range=[-4, 4], L=10).to(device)
    dir_encoder = PositionalEncoder(data_range=[-1, 1], L=4).to(device)
    nerf = NeRF(
        xyz_encoder=xyz_encoder,
        dir_encoder=dir_encoder,
        input_dim=60,
        view_dim=24,
        num_layers=8,
        hidden_dim=256,
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    # train the model
    N_samples = 64 # number of samples per ray
    N_rand = 1024 # number of rays per iteration, adjust according to your GPU memory (1024 is too much)
    for epoch in tqdm(range(epochs+1)):
        for i in range(len(dataset)):
            img, pose, focal = dataset[i]
            img = img.to(device)
            H, W = img.shape[:2]
            pose = pose.to(device)
            focal = focal.to(device)

            # sample rays
            rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

            # print("rays_o.shape",rays_o.shape)
            # print("rays_d.shape",rays_d.shape)

            # random select N_rand rays to reduce memory usage
            selected_rays_o, selected_rays_d, selected_gt_rgb = random_select_rays(H, W, rays_o, rays_d, img, N_rand)

            # print("selected_rays_o.shape",selected_rays_o.shape)
            # print("selected_rays_d.shape",selected_rays_d.shape)
            # print("selected_gt_rgb.shape",selected_gt_rgb.shape)

            # volumn render
            pred_rgb = volumn_render(NeRF=nerf, rays_o=selected_rays_o, rays_d=selected_rays_d, N_samples=N_samples)


            # print("pred_rgb.shape",pred_rgb.shape)
            # print("selected_gt_rgb.shape",selected_gt_rgb.shape)

            # import pdb; pdb.set_trace()
            l = loss(pred_rgb, selected_gt_rgb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            psnr_value = psnr(selected_gt_rgb.detach().cpu().numpy(), pred_rgb.detach().cpu().numpy(), data_range=1)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {l.item()}, PSNR: {psnr_value}')
            with torch.no_grad():
                chunk_size = 1024 # adjust according to your GPU memory
                pred_rgb = []
                for i in range(0, H*W, chunk_size):
                    rays_o_chunk = rays_o.reshape(-1, 3)[i:i+chunk_size]
                    rays_d_chunk = rays_d.reshape(-1, 3)[i:i+chunk_size]
                    pred_rgb.append(volumn_render(NeRF=nerf, rays_o=rays_o_chunk, rays_d=rays_d_chunk, N_samples=N_samples))
                pred_rgb = torch.cat(pred_rgb, dim=0)
                torchvision.utils.save_image(pred_rgb.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'./output/NeRF3D/pred_{epoch}.png')
                torchvision.utils.save_image(img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'./output/NeRF3D/gt_{epoch}.png')


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    data_path = './data/lego' # data path
    psnr_value = fit_images_and_calculate_psnr(data_path)
