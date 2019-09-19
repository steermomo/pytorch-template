import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from data_loader import make_data_loader
from model.model import Unet_resnet50, DeeplabV3Resnet, FPN_Resnet34
import cv2
import gc
import pickle
from multiprocessing import Pool
from itertools import repeat
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_model_result():
    ckpt_path = './FPN_resnet34_sigmoid_valdice_0.9431.pth'
    model = FPN_Resnet34(nclass=4)
    model.load_state_dict(torch.load(ckpt_path))

    train_loader, val_loader, test_loader, nclass = \
        make_data_loader(dataset='steel', data_dir='/data2/hangli_data/kaggle/steel/data',
                         batch_size=36, shuffle=True, num_workers=12,
                         training=True, base_size=[256, 1600], crop_size=[256, 1600], NUM_CLASSES=4)

    pred_ret = []
    target_ret = []
    # tbar = val_loader
    tbar = tqdm(val_loader, ascii=True)
    model = model.cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    for batch_idx, sample in enumerate(tbar):
        data, target = sample['image'], sample['label']
        with torch.no_grad():
            data = data.cuda()
            pred = model(data)
            # pred = torch.sigmoid(pred)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        pred_ret.append(pred)
        target_ret.append(target)

    pred_ret = np.concatenate(pred_ret, axis=0)
    target_ret = np.concatenate(target_ret, axis=0)
    return pred_ret, target_ret


def remove_small_one(predict, min_size):
    H, W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H, W), np.bool)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = True
    del component
    return predict


def remove_small(predict, min_size):
    tbar = tqdm(range(len(predict)))
    tbar.set_description('remove small')
    for b in tbar:
        for c in range(4):
            predict[b, c] = remove_small_one(predict[b, c], min_size[c])
    return predict


def compute_metric(truth, predict):

    num = len(truth)
    # t = truth.reshape(num * 4, -1).astype(np.float32)
    # p = predict.reshape(num * 4, -1).astype(np.float32)
    t = truth.reshape(num * 4, -1).astype(np.int8)
    p = predict.reshape(num * 4, -1).astype(np.int8)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    # hit pred
    h_neg = (p_sum == 0).astype(np.float32)
    h_pos = (p_sum > 0).astype(np.float32)
    gc.collect()
    d_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-12)

    print(f'dice: {np.mean(d_pos)}')

    t_sum = t_sum.reshape(num, 4)
    p_sum = p_sum.reshape(num, 4)
    h_neg = h_neg.reshape(num, 4)
    h_pos = h_pos.reshape(num, 4)
    d_pos = d_pos.reshape(num, 4)

    gc.collect()
    #for each class
    hit_neg = []
    hit_pos = []
    dice_pos = []
    for c in range(4):
        neg_index = np.where(t_sum[:, c] == 0)[0]
        pos_index = np.where(t_sum[:, c] >= 1)[0]
        hit_neg.append(h_neg[:, c][neg_index])
        hit_pos.append(h_pos[:, c][pos_index])
        dice_pos.append(d_pos[:, c][pos_index])

    ##
    hit_neg_all = np.concatenate(hit_neg).mean()
    hit_pos_all = np.concatenate(hit_pos).mean()
    hit_neg = [np.nan_to_num(h.mean(), 0) for h in hit_neg]
    hit_pos = [np.nan_to_num(h.mean(), 0) for h in hit_pos]
    dice_pos = [np.nan_to_num(d.mean(), 0) for d in dice_pos]

    ## from kaggle probing ...
    kaggle_pos = np.array([128, 43, 741, 120])
    kaggle_neg_all = 6172
    kaggle_all = 1801 * 4
    kaggle = (hit_neg_all * kaggle_neg_all + sum(dice_pos * kaggle_pos)) / kaggle_all

    # #confusion matrix
    # t = truth.transpose(1, 0, 2, 3).reshape(4, -1)
    # t = np.vstack([t.sum(0, keepdims=True) == 0, t])
    # p = predict.transpose(0, 2, 3, 1).reshape(-1, 4)
    # p = np.hstack([p.sum(1, keepdims=True) == 0, p])

    # confusion = np.zeros((5, 5), np.float32)
    # for c in range(5):
    #     index = np.where(t[c] == 1)[0]
    #     confusion[c] = p[index].sum(0) / len(index)

    #print (np.array_str(confusion, precision=3, suppress_small=True))
    return kaggle, hit_neg_all, hit_pos_all, hit_neg, hit_pos, dice_pos,  # confusion


def dice_channel_np(probability, truth, threshold=0.5):
    batch_size = probability.shape[0]
    channel_num = probability.shape[1]
    if channel_num == 5:
        st = 1
        ed = 5
    else:
        st = 0
        ed = 4
    mean_dice_channel = 0.
    tbar = tqdm(range(batch_size))
    
    for i in tbar:
        for j in range(st, ed):
            if channel_num == 5:
                truth_map = (truth[i, ...] == j).astype('uint8')
            else:
                truth_map = truth[i, j, ...]

            channel_dice = dice_single_channel(probability[i, j, :, :], truth_map, threshold)
            tbar.set_description(f'processing dice {channel_dice:.3f}')
            mean_dice_channel += channel_dice / (batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps=1E-9):
    p = (probability.reshape(-1) > threshold).astype('float')
    # p = probability.reshape(-1).astype('float')
    t = (truth.reshape(-1) > 0.5).astype('float')
    dice = (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice

def dice_channel(probability, truth, channel, threshold=0.5):
    batch_size = probability.shape[0]
    mean_dice_channel = 0.
    # tbar = tqdm(range(batch_size))
    for i in range(batch_size):
        truth_map = truth[i, channel, ...]
        channel_dice = dice_single_channel(probability[i, channel, :, :], truth_map, threshold)
        # tbar.set_description(f'processing dice {channel_dice:.3f}')
        mean_dice_channel += channel_dice / (batch_size)
    return mean_dice_channel

def main():
    pred, target = get_model_result()   
    threshold_pixel = [0.50, 0.50, 0.50, 0.50, ]
    threshold_pixel = [0.0, 0.0, 0.0, 0.0, ]
    pred = pred > np.array(threshold_pixel).reshape(-1, 4, 1, 1).astype(np.uint8)
    print(f'pred shape:{pred.shape} target shape:{target.shape}')
    gc.collect()
    logfile = open('prob_log.txt', 'wt')
    bst_dice = 0
    bst_param = None
    md = []
    
    with Pool(6) as p:
        rt = p.imap_unordered(dice_warp, zip(repeat(pred), repeat(target), md))
    # with open('rt_dump', 'wb') as outfile:
    #     pickle.dump(rt, outfile)
    
    # sort_rt = sorted(rt, key=lambda x: x[0], reverse=True)
    # bst_dice = sort_rt[0][0]
    # bst_param = sort_rt[0][1]
    bst_dice = 0
    bst_param = None
    for dice, param in rt:
        print(f'dice: {dice} => {param}', file=logfile)
        if bst_dice < dice:
            bst_param = param
    print(f'bst dice {bst_dice} param {bst_param}')
    print(f'bst dice {bst_dice} param {bst_param}', file=logfile)
    logfile.close()


if __name__ == "__main__":
    main()
