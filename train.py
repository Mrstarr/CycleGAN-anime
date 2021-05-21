"""Training script for GAN"""

from __future__ import print_function

from Config import CONFIG
from model import Generator, Discriminator
from datetime import datetime
from itertools import chain
from scheduler import SchedulerFactory
from datasets import LoadData
import os
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import wandb


def init_normal(m):
    """A function for weights initialization"""
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def train(config, device):
    """
    Train our networks.
    Args:
        config: configuration
        device: The device to train on
    """

    wandb.init(project="CycleGAN_DD2424")

    # Init models
    Gen_A2B = Generator(input_nc=config.input_ch, output_nc=config.output_ch).to(device)
    Gen_B2A = Generator(input_nc=config.input_ch, output_nc=config.output_ch).to(device)
    Dis_A = Discriminator(input_nc=config.input_ch).to(device)
    Dis_B = Discriminator(input_nc=config.input_ch).to(device)

    Gen_A2B.apply(init_normal)
    Gen_B2A.apply(init_normal)
    Dis_A.apply(init_normal)
    Dis_B.apply(init_normal)

    wandb.watch(Gen_A2B)
    wandb.watch(Gen_B2A)
    wandb.watch(Dis_A)
    wandb.watch(Dis_B)

    # Set wandb and training params
    lr = wandb.config.learning_rate = config.learning_rate
    wandb.config.time_string = time_string = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )   
    run_name = wandb.config.run_name = "saved_models/det_{}".format(time_string)
    w_gan = wandb.config.w_gan = config.w_gan
    w_cycle = wandb.config.w_cycle = config.w_cycle
    w_identity = wandb.config.w_identity = config.w_identity

    # Init optimizer
    opt_Gen = torch.optim.Adam(
        params=chain(Gen_A2B.parameters(), Gen_B2A.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    # omit: betas=(0.5, 0.999), default: (0.9, 0.999)
    opt_DisA = torch.optim.Adam(params=Dis_A.parameters(), lr=lr, betas=(0.5, 0.999))  # ditto
    opt_DisB = torch.optim.Adam(params=Dis_B.parameters(), lr=lr, betas=(0.5, 0.999))  # ditto

    # Init lr scheduler
    scheduler_Gen = SchedulerFactory(config, "linear", opt_Gen)()
    scheduler_DisA = SchedulerFactory(config, "linear", opt_DisA)()
    scheduler_DisB = SchedulerFactory(config, "linear", opt_DisB)()

    # Predefine target tensor for each batch
    target_1 = torch.ones(config.batch_size)
    target_0 = torch.zeros(config.batch_size)

    # Dataloader
    train_loader = LoadData().train()

    print("Training started...")

    # ----- epoch loop -----#
    # 'i_' indicates this is an index.
    for i_epoch in range(config.start_epoch, config.num_epoch):

        # ----- batch loop -----#
        for i_img, img_batch in enumerate(train_loader):
            # img_batch: dict; keys: 'A', 'B',
            # values: torch.tensor （shape: batch X 3 X H X W)
            # How to show image when debugging:
            # >>> plt.imshow(img_batch['A'].squeeze(0).permute(1, 2, 0))
            # >>> plt.show()
            opt_Gen.zero_grad()
            # ----- STEP 0: Get real images ----- #
            real_A = img_batch["A"].to(device)
            real_B = img_batch["B"].to(device)

            # ----- STEP 1: Train generators ----- #

            # forward
            identity_A = Gen_B2A(real_A)  # A -> A
            identity_B = Gen_A2B(real_B)
            fake_A = Gen_B2A(real_B)  # B -> A
            fake_B = Gen_A2B(real_A)
            rec_A = Gen_B2A(fake_B)  # A -> B -> A
            rec_B = Gen_A2B(fake_A)

            loss_identity = F.l1_loss(real_A, identity_A) + F.l1_loss(
                real_B, identity_B
            )
            loss_GAN = F.mse_loss(Dis_B(fake_B), target_1) + F.mse_loss(
                Dis_A(fake_A), target_1
            )
            loss_cycle = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)

            loss_G = (
                loss_GAN * w_gan + loss_cycle * w_cycle + loss_identity * w_identity
            )

            # backward
            
            loss_G.backward()
            opt_Gen.step()
            # ----- Finish training generators ----- #

            # ----- STEP 2: Train discriminators ----- #
            opt_DisA.zero_grad()
            # forward: A
            loss_A_real = F.mse_loss(Dis_A(real_A), target_1)
            # warning: must detach() fake_A to exclude the G part in the graph
            loss_A_fake = F.mse_loss(Dis_A(fake_A.detach()), target_0)
            loss_D_A = (loss_A_real + loss_A_fake) * 0.5

            # backward: A
            
            loss_D_A.backward()
            opt_DisA.step()
            opt_DisB.zero_grad()
            # forward: B
            loss_B_real = F.mse_loss(Dis_B(real_B), target_1)
            loss_B_fake = F.mse_loss(Dis_B(fake_B.detach()), target_0)
            loss_D_B = (loss_B_real + loss_B_fake) * 0.5

            # backward: B
            
            loss_D_B.backward()
            opt_DisB.step()

            # ----- Finish training discriminators ----- #

            # after training, log data to wandb
            wandb.log(
                {
                    "Loss G": loss_G.item(),
                    "Loss G GAN": loss_GAN.item(),
                    "loss G cycle": loss_cycle.item(),
                    "loss G identity": loss_identity.item(),
                    "loss D": (loss_D_A + loss_D_B).item(),
                    "loss D A": loss_D_A.item(),
                    "loss D B": loss_D_B.item(),
                },
                # step=current_iteration,
            )
            print("Loss_G:",loss_G.item()," Loss_D:",(loss_D_A + loss_D_B).item())

        # ----- end of batch loop ----- #
        # ----- back to epoch loop ----- #

        # Update learning rates
        scheduler_Gen.step()
        scheduler_DisA.step()
        scheduler_DisB.step()

        # log last image in each epoch
        wandb.log(
            {
                "real_A": wandb.Image(real_A),
                "real_B": wandb.Image(real_B),
                "fake_A": wandb.Image(fake_A),
                "fake_B": wandb.Image(fake_B),
            },
            step=i_epoch,
        )

        # Save models checkpoints
        save_path = "output/{}".format(run_name)
        os.mkdir(save_path)
        torch.save(
            Gen_A2B.state_dict(), save_path + "/Gen_A2B_epoch_{}.pth".format(i_epoch)
        )
        torch.save(
            Gen_B2A.state_dict(), save_path + "/Gen_B2A_epoch_{}.pth".format(i_epoch)
        )
        torch.save(
            Dis_A.state_dict(), save_path + "/Dis_A_epoch_{}.pth".format(i_epoch)
        )
        torch.save(
            Dis_B.state_dict(), save_path + "/Dis_B_epoch_{}.pth".format(i_epoch)
        )
        print("Model weights saved at {}".format(save_path))


if __name__ == "__main__":
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(config=CONFIG, device=current_device)
