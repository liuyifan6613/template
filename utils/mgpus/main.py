# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
from torchvision.utils import save_image

from accelerate import Accelerator, DataLoaderConfiguration
import torch.nn as nn


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a ResNet50 on the Oxford-IIT Pet Dataset
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


def ImageTransform(loadSize):
    return {"train": Compose([
        Resize((loadSize, loadSize)),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        Resize((loadSize, loadSize)),
        ToTensor(),
    ])}


class BasicData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = os.listdir(path_gt)
        self.data_img = os.listdir(path_img)
        self.data_img.sort()
        self.data_gt.sort()
        self.loadSize = loadSize
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):
        gt = Image.open(os.path.join(self.path_gt, self.data_img[idx]))
        img = Image.open(os.path.join(self.path_img, self.data_img[idx]))
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
        return {"image": img, "label": gt, "name": name}

def psnr(img1, img2):
    mse = torch.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return np.array((20 * torch.log10(max_pixel / torch.sqrt(mse))).detach().cpu())

def training_function(args, model, train_dataset, eval_dataset, loss_function):
    # Initialize accelerator
    dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=args.use_stateful_dataloader)
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            log_with="all",
            project_dir=args.project_dir,
            dataloader_config=dataloader_config,
        )
    else:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, dataloader_config=dataloader_config
        )

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = args.lr
    num_epochs = args.num_epochs
    seed = args.seed
    batch_size = args.batch_size
    image_size = args.image_size
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)

    # Parse out whether we are saving every epoch or after a certain number of batches
    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, vars(args))

    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=1, num_workers=4)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Freezing the base model
    for param in model.parameters():
        param.requires_grad = True

    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr / 25)

    # Instantiate learning rate scheduler
    lr_scheduler = ConstantLR(optimizer=optimizer, factor=0.5, total_iters=100)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the starting epoch so files are named properly
    starting_epoch = 0

    best_psnr = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # Now we train the model
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_dataloader
        for batch in active_dataloader:
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            for k in batch.keys():
                try:
                    batch[k] = batch[k].to(accelerator.device)
                except:
                    pass
            inputs = batch["image"]
            outputs = model(inputs)
            loss = loss_function(outputs, batch["label"])
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            overall_step += 1
            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
        model.eval()
        eval_metric = []
        if epoch % args.eval_every == 0:
            output_dir = f"model_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            for step, batch in enumerate(eval_dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                for k in batch.keys():
                    try:
                        batch[k] = batch[k].to(accelerator.device)
                    except:
                        pass
                inputs = batch["image"]
                name = batch['name']
                with torch.no_grad():
                    outputs = model(inputs)
                predictions = outputs
                os.makedirs('./eval_res', exist_ok=True)
                save_image(outputs, os.path.join('./eval_res', name[0]))
                predictions, references = accelerator.gather_for_metrics((predictions, batch["label"]))
                for i in range(predictions.shape[0]):
                    eval_metric.append(psnr(predictions[i]*255.0, references[i]*255.0))
                if accelerator.is_main_process:
                    if np.mean(eval_metric)>best_psnr:
                        best_psnr = np.mean(eval_metric)
                        accelerator.save_state(output_dir.replace(f'model_{epoch}', 'best'))
            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch: {epoch}| PSNR: {best_psnr:.2f}| Loss: {total_loss.item() / len(train_dataloader):.5f}|")
        if args.with_tracking and accelerator.is_main_process:
            accelerator.log(
                {
                    "train_loss": total_loss.item() / len(train_dataloader),
                    # "epoch": epoch,
                    "best_psnr": best_psnr,
                },
                step=overall_step,
            )
    accelerator.end_training()


def test_function(args, model, eval_dataset):
    # Initialize accelerator
    dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=args.use_stateful_dataloader)
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            log_with="all",
            project_dir=args.project_dir,
            dataloader_config=dataloader_config,
        )
    else:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, dataloader_config=dataloader_config
        )

    seed = args.seed
    batch_size = args.batch_size
    image_size = args.image_size


    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Instantiate dataloaders.
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Freezing the base model
    for param in model.parameters():
        param.requires_grad = False

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, eval_dataloader, = accelerator.prepare(
        model, eval_dataloader,
    )

    best_psnr = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

    model.eval()
    eval_metric = []
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        for k in batch.keys():
            try:
                batch[k] = batch[k].to(accelerator.device)
            except:
                pass
        inputs = batch["image"]
        name = batch['name']
        with torch.no_grad():
            outputs = model(inputs)
        predictions = outputs
        os.makedirs('./eval_test', exist_ok=True)
        for i in range(outputs.shape[0]):
            save_image(outputs[i].unsqueeze(dim=0), os.path.join('./eval_test', name[i]))
        predictions, references = accelerator.gather_for_metrics((predictions, batch["label"]))
        for i in range(predictions.shape[0]):
            eval_metric.append(psnr(predictions[i]*255.0, references[i]*255.0))
        if accelerator.is_main_process:
            best_psnr = np.mean(eval_metric)
    # Use accelerator.print to print only on the main process.
    accelerator.print(f"PSNR: {best_psnr:.2f}|")
    accelerator.end_training()

class Simple3to3CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # 输出为3通道
        )

    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument('--train_img', type=str, default='/home/yfliu/Dataset/set14/',
                        help='Training image.')
    parser.add_argument('--train_gt', type=str, default='/home/yfliu/Dataset/set14/',
                        help='Training GroundTruth image.')
    parser.add_argument('--test_img', type=str, default='/home/yfliu/Dataset/set14/',
                        help='Test image.')
    parser.add_argument('--test_gt', type=str, default='/home/yfliu/Dataset/set14/',
                        help='Test GroundTruth image.')
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Frequency of evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--use_stateful_dataloader",
        action="store_true",
        help="If the dataloader should be a resumable stateful dataloader.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training and inference batch size.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="The size of images to resize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Selected seed.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=False,
        help="Train or test.",
    )
    args = parser.parse_args()

    model = Simple3to3CNN()
    test_dataset = BasicData(args.test_img, args.test_img, args.image_size)
    if args.test=='False':
        train_dataset = BasicData(args.train_img, args.train_gt, args.image_size)
        loss = torch.nn.MSELoss()
        training_function(args, model, train_dataset, test_dataset, loss)
    else:
        test_function(args, model, test_dataset)

if __name__ == "__main__":
    main()
