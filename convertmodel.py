import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from model.model import Unet_resnet50

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Using cuda
    model = Unet_resnet50(nclass=5)
    gpu_ids = [0]
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    # Resuming checkpoint
    # model_path = './saved/models/Kaggle_steel_Unet_resnet50/0819_151248/model_best.pth'
    model_path = './saved/models/Kaggle_steel_Unet_resnet50/0820_184252/checkpoint-epoch100.pth'

    checkpoint = torch.load(model_path)

    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    # model.module.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    torch.save(model.state_dict(), f'Unet_resnet50_e100_sigmoid.pth')

    # m = torch.load('deeplab.pth')
    # summary(m, input_size=(3, 256, 1600))


if __name__ == "__main__":
    main()
