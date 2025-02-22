import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
from typing import Tuple, Union, List
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from skimage.metrics import structural_similarity as ssim
import cv2


def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1_gray, img2_gray)


def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def calculate_metrics(folder1, folder2):
    images_folder1 = sorted(os.listdir(folder1))
    images_folder2 = sorted(os.listdir(folder2))

    if len(images_folder1) != len(images_folder2):
        print("Please check file numbers in two folders！")
        return

    psnr_values = []
    ssim_values = []

    for img_name1, img_name2 in zip(images_folder1, images_folder2):
        img_path1 = os.path.join(folder1, img_name1)
        img_path2 = os.path.join(folder2, img_name2)

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if (h1, w1) != (h2, w2):
            if h1 * w1 > h2 * w2:
                img1 = resize_image(img1, (w2, h2))
            else:
                img2 = resize_image(img2, (w1, h1))

        psnr_value = psnr(img1, img2)
        psnr_values.append(psnr_value)

        ssim_value = compute_ssim(img1, img2)
        ssim_values.append(ssim_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"\nAvg PSNR: {avg_psnr:.2f} dB")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


class Swish(nn.Module):
    """
    ### Swish activation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer

        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(x))
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        return x


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, ):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.dia1 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2))
        self.dia2 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4))
        self.dia3 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8))
        self.dia4 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16))
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.dia1(x)
        x = self.dia2(x)
        x = self.dia3(x)
        x = self.dia4(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, input_channels: int = 3, output_channels: int = 3, n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 n_blocks: int = 2, ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4))
            # Final block to reduce the number of channels
            in_channels = n_channels * (ch_mults[i - 1] if i >= 1 else 1)
            up.append(UpBlock(in_channels, out_channels, n_channels * 4))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                # print(x.shape, s.shape)
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        # Final normalization and convolution
        return self.final(self.act(x))


def init__result_Dir(work_dir):
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    path = os.path.join(work_dir, str(max_model))
    os.mkdir(path)
    return path


def ImageTransform(loadSize):
    return {"train": Compose([
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        ToTensor(),
    ])}


class BasicData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = os.listdir(path_gt)
        self.data_img = os.listdir(path_img)
        self.loadSize = loadSize
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):

        gt = Image.open(os.path.join(self.path_gt, self.data_img[idx])).resize((self.loadSize, self.loadSize))
        img = Image.open(os.path.join(self.path_img, self.data_img[idx])).resize((self.loadSize, self.loadSize))
        img = img.convert('RGB')
        gt = gt.convert('RGB')
        if self.mode == 1:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.ImgTrans[0](img)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            img = self.ImgTrans(img)
            gt = self.ImgTrans(gt)
        name = self.data_img[idx]
        return img, gt, name


class Trainer:
    def __init__(self, args):
        self.mode = args.mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = args.channel_in
        out_channels = args.channel_out
        self.out_channels = out_channels
        self.network = UNet(
            input_channels=in_channels,
            output_channels=out_channels,
            n_channels=args.model_channels,
            ch_mults=[1, 2, 3, 4],
            n_blocks=1,
        ).to(self.device)
        self.test_img_save_path = args.test_img_save_path
        if not os.path.exists(self.test_img_save_path):
            os.makedirs(self.test_img_save_path)
        self.epoch = args.continue_training_epochs
        self.train_gt = args.train_gt
        self.train_img = args.train_img
        self.save_model_every = args.save_model_every
        self.weight_save_path = args.weight_save_path
        self.test_img = args.test_img
        self.test_gt = args.test_gt
        self.image_size = args.train_image_size
        self.max_epoch = args.max_epoch
        self.learning_rate = args.learning_rate
        self.test_every = args.test_every
        self.best_psnr = 0.0
        self.best_epoch = 0.0
        self.work_dir = args.work_dir

        if args.checkpoint:
            self.network.load_state_dict(torch.load(args.checkpoint))

        dataset_train = BasicData(self.train_img, self.train_gt, self.image_size, self.mode)
        self.batch_size = args.batch_size
        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                           num_workers=8)

        dataset_test = BasicData(self.test_img, self.test_gt, self.image_size, self.mode)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                          drop_last=False,
                                          num_workers=8)
        if args.loss == 'L1':
            self.loss = nn.L1Loss()
        elif args.loss == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')
            self.loss = nn.MSELoss()

    def test(self):
        with torch.no_grad():
            self.network.eval()
            tq = tqdm(self.dataloader_test)
            iteration = 0
            for img, gt, name in tq:
                tq.set_description(f'Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                iteration += 1
                predict = self.network(img.to(self.device))

                save_image(predict.cpu(), os.path.join(
                    self.test_img_save_path, f"{name[0]}"), nrow=4)

    def train(self):
        optimizer = optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        save_img_path = init__result_Dir(self.work_dir)
        print('Starting Training', f"Epoch is {self.epoch}")
        while self.epoch < self.max_epoch:
            iteration = 0
            loss = 0
            tq = tqdm(self.dataloader_train)
            for img, gt, name in tq:
                tq.set_description(
                    f'Epoch-Iteration {self.epoch} / {iteration} | Best PSNR {self.best_psnr:2f}| Loss {loss:5f}')
                iteration += 1
                self.network.train()
                optimizer.zero_grad()
                predict = self.network(img.to(self.device))
                pixel_loss = self.loss(predict, gt.to(self.device))
                loss = pixel_loss
                loss.backward()
                optimizer.step()
            if self.epoch % self.save_model_every == 0:
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
                img_save = torch.cat([img, gt, predict.cpu()], dim=3)
                save_image(img_save, os.path.join(
                    save_img_path, f"{self.epoch}.png"), nrow=4)
            print('Saving models')
            if not os.path.exists(self.weight_save_path):
                os.makedirs(self.weight_save_path)
            if self.epoch % self.save_model_every == 0:
                torch.save(self.network.state_dict(),
                           os.path.join(self.weight_save_path, f'model_{self.epoch}.pth'))
            self.epoch += 1
            if self.epoch % self.test_every == 0:
                with torch.no_grad():
                    self.network.eval()
                    tq_test = tqdm(self.dataloader_test)
                    iteration = 0
                    for img, gt, name in tq_test:
                        tq_test.set_description(f'Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                        iteration += 1
                        predict = self.network(img.to(self.device))
                        save_image(predict.cpu(), os.path.join(
                            self.test_img_save_path, f"{name[0]}"), nrow=4)
                psnr, _ = calculate_metrics(self.test_img_save_path, self.test_gt)
                if psnr > self.best_psnr:
                    torch.save(self.network.state_dict(),
                               os.path.join(self.weight_save_path, 'best.pth'))
                    self.best_psnr = psnr
                    self.best_epoch = self.best_epoch - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img', type=str,
                        default='./train_img')
    parser.add_argument('--train_gt', type=str, default='./train_gt')
    parser.add_argument('--test_img', type=str, default='./test_img')
    parser.add_argument('--test_gt', type=str, default='./test_gt')
    parser.add_argument('--checkpoint', type=str, default='。/model.pth')
    parser.add_argument('--train_image_size', type=int, default=512)
    parser.add_argument('--continue_training_epochs', type=int, default=0)
    parser.add_argument('--save_model_every', type=int, default=1)
    parser.add_argument('--test_every', type=str, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--loss', type=str, default='L2')

    parser.add_argument('--channel_in', type=int, default=3)
    parser.add_argument('--channel_out', type=int, default=3)
    parser.add_argument('--model_channels', type=int, default=32)

    parser.add_argument('--test_img_save_path', type=str, default='./test')
    parser.add_argument('--weight_save_path', type=str, default='./checkpoint')
    parser.add_argument('--work_dir', type=str, default='./Training')

    args = parser.parse_args()
    mode = args.mode
    runner = Trainer(args)
    if mode == 'train':
        runner.train()
    else:
        runner.test()


if __name__ == '__main__':
    main()
