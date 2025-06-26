import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from collections import OrderedDict
from torch.nn import functional as F
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import math
import torch.distributed as dist

opt = {
    # ----------------- options for synthesizing training data in RealESRGANModel ----------------- #
    'scale': 1,
    'gt_size': 200,
    
    # USM the ground-truth
    'l1_gt_usm': True,
    'percep_gt_usm': True,
    'gan_gt_usm': False,

    # the first degradation process
    'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
    'resize_range': [0.15, 1.5],
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 30],
    'poisson_scale_range': [0.05, 3],
    'gray_noise_prob': 0.4,
    'jpeg_range': [30, 95],

    # the second degradation process
    'second_blur_prob': 0.8,
    'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
    'resize_range2': [0.3, 1.2],
    'gaussian_noise_prob2': 0.5,
    'noise_range2': [1, 25],
    'poisson_scale_range2': [0.05, 2.5],
    'gray_noise_prob2': 0.4,
    'jpeg_range2': [30, 95],

    # dataset and data loader settings

    'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob': 0.1,
    'blur_sigma': [0.2, 3],
    'betag_range': [0.5, 4],
    'betap_range': [1, 2],
    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    'final_sinc_prob': 0.8,
    'use_hflip': True,
    'use_rot': False,
}

class Degradation():
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self):
        super(Degradation, self).__init__()
        self.opt = opt
        # rank = dist.get_rank()
        # self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1


        self.jpeger = DiffJPEG(differentiable=False).to(self.device)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().to(self.device)   # do usm sharpening

    @torch.no_grad()
    def feed_data(self, gt):
        """
            Input: [B, C, H, W], RGB, range from 0 to 1.
            Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        data = self.get_kernel()
        self.gt = gt.to(self.device)
        self.gt_usm = self.usm_sharpener(self.gt)

        self.kernel1 = data['kernel1'].to(self.device)
        self.kernel2 = data['kernel2'].to(self.device)
        self.sinc_kernel = data['sinc_kernel'].to(self.device)

        ori_h, ori_w = self.gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(self.gt_usm, self.kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
        # add noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = self.opt['gt_size']
        (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                self.opt['scale'])
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        return self.lq, self.gt


    def get_kernel(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        return return_d

if __name__ == '__main__':
    test_img = './comic.png'
    from PIL import Image
    test_img = Image.open(test_img).convert('RGB')
    test_img = torch.from_numpy(np.array(test_img)).permute(2, 0, 1).float() / 255.
    test_img = test_img.unsqueeze(0)
    degradation = Degradation()
    lq, gt = degradation.feed_data(test_img)
    import torchvision
    torchvision.utils.save_image(lq, 'lq.png')
    torchvision.utils.save_image(gt, 'gt.png')
