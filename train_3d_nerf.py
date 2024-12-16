import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
import tensorflow as tf
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
        self.data_range = data_range[1]
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table(self.L, self.data_range))

    def get_sinusoid_encoding_table(self, L, data_range):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / data_range) for hid_j in range(data_range)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(L + 1)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, x.size(1)].clone().detach()

# Hash grid encoding
class MultiResHashGridEncoder(nn.Module):
    """
    Implement the Hash Grid Encoding function.
    Please note that the input tensor x should be normalized to the range [0, 1].

    Args:
    x (torch.Tensor): The input tensor to be embedded.
    n_levels (int): The number of grid levels.
    n_features_per_level (int): The number of features per grid level.

    Returns:
    torch.Tensor: The embedded tensor.
    """
    def __init__(self, data_range, n_levels, n_features_per_level):
        super(MultiResHashGridEncoder, self).__init__()
        pass



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
    # Convert pixel coordinates to camera coordinates
    # Transform camera directions to world directions using the camera-to-world matrix
    # The origin of each ray is the camera position

    #H = H.cpu()
    #W = W.cpu()
    f = f.cpu()
    #c2w = c2w.cpu()
    '''
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / f, -(j - H * .5) / f, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    '''
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / f, -(j - H * .5) / f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d

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
    # Generate uniform samples in the interval [0, 1] for each segment
    # Add the random jitter to the uniform samples
    # Calculate the points along the ray by interpolating between tn and tf
    # using the sampled values

    t_vals = tf.linspace(0., 1., N_samples)
    z_vals = tn * (1. - t_vals) + tf * (t_vals)

    return z_vals

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
    # Sample points along each ray, from near plane to far plane

    # Calculate the points along the rays by sampling
    # pts.shape => (N_rays, N_samples, 3)

    # Get the color and density from the NeRF model

    # Volume rendering: compute the transmittance and accumulate the color
    # alpha = 1. - torch.exp(-delta * torch.relu(sigma)) # original formula in the paper
    # alpha = 1. - torch.exp(-delta * torch.nn.Softplus()(sigma)) # you can choose the trick to stabilize training

    # Compute the weights for each sample using alpha compositing

    # Accumulate the color along each ray
    z_vals = sample_points_along_the_ray(rays_o, rays_o + rays_d*100, N_samples)
    pred_rgb = NeRF(z_vals)

    return pred_rgb



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
    selected_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = torch.tensor(selected_img, dtype=torch.float32)  # Ensure float32 dtype

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
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    # train the model
    N_samples = 64 # number of samples per ray
    N_rand = 1024 # number of rays per iteration, adjust according to your GPU memory
    for epoch in tqdm(range(epochs+1)):
        for i in range(len(dataset)):
            img, pose, focal = dataset[i]
            img = img.to(device)
            H, W = img.shape[:2]
            pose = pose.to(device)
            focal = focal.to(device)

            # sample rays
            rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

            # random select N_rand rays to reduce memory usage
            selected_rays_o, selected_rays_d, selected_gt_rgb = random_select_rays(H, W, rays_o, rays_d, img, N_rand)

            # volumn render
            pred_rgb = volumn_render(NeRF=nerf, rays_o=selected_rays_o, rays_d=selected_rays_d, N_samples=N_samples)

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
                torchvision.utils.save_image(pred_rgb.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'output/NeRF/pred_{epoch}.png')
                torchvision.utils.save_image(img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'output/NeRF/gt_{epoch}.png')


if __name__ == '__main__':
    data_path = './data/lego' # data path
    psnr_value = fit_images_and_calculate_psnr(data_path)
