import glob
import random
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as trans
from Config import CONFIG

"""
>>> from datasets import LoadData
>>> train_loader = LoadData().train()
>>> test_loader = LoadData().test()
"""


class LoadData():
    def train(self):
        return DataLoader(ImageData(CONFIG.data_root),
                          batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_cpu)

    def test(self):
        return DataLoader(ImageData(CONFIG.data_root, train=False),
                          batch_size=CONFIG.batch_size, num_workers=CONFIG.num_cpu)


class ImageData(Dataset):
    def __init__(self, root, train=True):
        if train:
            self.transform = trans.Compose([trans.RandomHorizontalFlip(),
                                            trans.Resize((int(CONFIG.size_h * 1.12), int(CONFIG.size_w * 1.12)), Image.BICUBIC),
                                            trans.RandomCrop((CONFIG.size_h, CONFIG.size_w)),
                                            trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.A_file = sorted(glob.glob(os.path.join(root, 'train/A') + '/*.*'))
            self.B_file = sorted(glob.glob(os.path.join(root, 'train/B') + '/*.*'))
        else:
            self.transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.A_file = sorted(glob.glob(os.path.join(root, 'test/A') + '/*.*'))
            self.B_file = sorted(glob.glob(os.path.join(root, 'test/B') + '/*.*'))

    def __getitem__(self, index):
        A = self.transform(Image.open(self.A_file[index % len(self.A_file)]))
        B = self.transform(Image.open(self.B_file[random.randint(0, len(self.B_file) - 1)]))

        return {'A': A, 'B': B}

    def __len__(self):
        return max(len(self.A_file), len(self.B_file))