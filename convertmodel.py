import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from model.model import Unet_resnet50, DeeplabV3Resnet, FPN_Resnet34

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Using cuda
    # model = Unet_resnet50(nclass=4)
    # model = DeeplabV3Resnet(nclass=5)
    model = FPN_Resnet34(nclass=4)
    gpu_ids = [0]
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    # Resuming checkpoint
    # model_path = './saved/models/Kaggle_steel_Unet_resnet50/0819_151248/model_best.pth'
    # model_path = './saved/models/Kaggle_steel_Unet_resnet50/0820_184252/checkpoint-epoch100.pth'
    model_path = './saved/models/Kaggle_steel_Unet_resnet50/0903_192320/checkpoint-epoch63.pth'
    model_path = './saved/models/Kaggle_steel_Unet_resnet50/0904_201748/checkpoint-epoch125.pth'
    # model_path = './saved/Kaggle_steel_deeplabv3_resnet/models/0908_231439/checkpoint-epoch196.pth'
    model_path = './saved/Kaggle_steel_deeplabv3_resnet/models/0912_150339/checkpoint-epoch196.pth'
    model_path = './saved/Kaggle_steel_FPN_resnet34/models/0915_160944/checkpoint-epoch109.pth'
    checkpoint = torch.load(model_path)

    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    # model.module.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    torch.save(model.module.state_dict(), f'FPN_resnet34_sigmoid_valdice_0.9431.pth')

    # m = torch.load('deeplab.pth')
    # summary(m, input_size=(3, 256, 1600))


if __name__ == "__main__":
    main()
