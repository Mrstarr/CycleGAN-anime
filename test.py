"""Test script for CycleGAN."""
import os
import torch
from torchvision.utils import save_image
from model import Generator
from datasets import LoadData
from Config import CONFIG


def test(config, device):
    """
    Test our networks.
    Args:
        config: configuration
        device: The device to test on
    """

    Gen_A2B = Generator(input_nc=config.input_ch, output_nc=config.output_ch).to(device)
    Gen_B2A = Generator(input_nc=config.input_ch, output_nc=config.output_ch).to(device)
    Gen_A2B.load_state_dict(torch.load(config.path_a2b))
    Gen_B2A.load_state_dict(torch.load(config.path_b2a))
    Gen_A2B.eval()
    Gen_B2A.eval()

    test_loader = LoadData().test()

    path_A = 'output/result_A'
    path_B = 'output/result_B'

    if not os.path.exists(path_A):
        os.makedirs(path_A)
    if not os.path.exists(path_B):
        os.makedirs(path_B)

    print("Testing started...")
    for i_img, img_batch in enumerate(test_loader):

        real_A = img_batch['A'].to(device)
        real_B = img_batch['B'].to(device)
        fake_B = Gen_A2B(real_A)
        fake_A = Gen_B2A(real_B)

        output_A = torch.mul(torch.add(fake_A, 1.0), 0.5)
        output_B = torch.mul(torch.add(fake_B, 1.0), 0.5)

        for d in range(output_A.shape[0]):
            save_A = output_A[d]
            save_B = output_B[d]
            save_image(save_A, path_A + '/{}_{}.png'.format(i_img, d))
            save_image(save_B, path_B + '/{}_{}.png'.format(i_img, d))


if __name__ == "__main__":
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(config=CONFIG, device=current_device)
