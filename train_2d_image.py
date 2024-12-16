import torch
from torch import nn
import torchvision
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from nerf_model import NeRF2D

class PositionalEncoder(nn.Module):
    """
    Implement the Position Encoding function.
    Defines a function that embeds x to (sin(2^k*pi*x), cos(2^k*pi*x), ...)

    Args:
    x (torch.Tensor): The input tensor to be embedded.
    L (int): The number of levels to embed.

    Returns:
    torch.Tensor: The embedded tensor.
    """
    def __init__(self, L):
        super(PositionalEncoder, self).__init__()
        self.L = L
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table(self.L))

    def get_sinusoid_encoding_table(self, L):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / 2) for hid_j in range(2)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(L + 1)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, x.size(1)].clone().detach()

# Main function
def fit_image_and_calculate_psnr(image_path, epochs=2000, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load image
    image = Image.open(image_path).resize((910//4, 510//4))
    image_array = np.array(image)
    img = torch.tensor(image_array, dtype=torch.float32) / 255.0
    H, W = img.shape[:2]
    img = img.reshape(-1, 3).to(device)

    ## get coordinates of each pixel
    x_coords = torch.linspace(0, H - 1, H) / H
    y_coords = torch.linspace(0, W - 1, W) / W
    x_coords, y_coords = torch.meshgrid(x_coords, y_coords)
    xy_coords = torch.stack((x_coords, y_coords), dim=-1).float().reshape(-1, 2)

    # define the model and optimizer
    # nerf_2d = NeRF2D(
    #     encoder=None, # without encoding, TODO: finish the encoder function in nerf_model.py
    #     input_dim=2,
    # ).to(device)
    encoder = PositionalEncoder(L=10).to(device) # TODO: fill in the number of levels for the PositionalEncoder
    nerf_2d = NeRF2D(
        encoder=encoder, # with position encoding
        input_dim=2, # TODO: fill in the input dimension for the NeRF2D model
    ).to(device)
    optimizer = torch.optim.Adam(nerf_2d.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # train the model
    for epoch in tqdm(range(epochs+1)):
        optimizer.zero_grad()
        pred_rgb = nerf_2d(xy_coords.to(device))
        loss = criterion(pred_rgb, img)
        loss.backward()
        optimizer.step()
        psnr_value = psnr(img.detach().cpu().numpy(), pred_rgb.detach().cpu().numpy(), data_range=1)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, PSNR: {psnr_value}')
            torchvision.utils.save_image(pred_rgb.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'./output/NeRF2D/pred_{epoch}.png')
            torchvision.utils.save_image(img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'./output/NeRF2D/gt_{epoch}.png')

if __name__ == '__main__':
    image_path = './data/Red_Bird_from_HKUST.jpg' # image path
    psnr_value = fit_image_and_calculate_psnr(image_path)
