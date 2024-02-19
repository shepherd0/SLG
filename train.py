import json

import torch
from thop import profile
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils.metrics import evaluate
from opt import opt
from utils.comm import generate_model
from utils.loss import DeepSupervisionLoss,  BceDiceLoss, FocalLoss, DiceLoss, DiceBCELoss
from utils.metrics import Metrics
from torch.autograd import Variable

import torchsummary

import matplotlib.pyplot as plt

import os.path
import time


def valid(model, valid_dataloader, total_batch):

    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'FPR', 'precision', 'F1', 'ZSI',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])


    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _FPR, _precision, _F1, _ZSI, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, FPR= _FPR, precision= _precision,
                            F1= _F1, ZSI= _ZSI, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )




    metrics_result = metrics.mean(total_batch)
    return metrics_result

def get_model_summary(model, input_size):
    device = torch.device("cuda" if opt.use_gpu else "cpu")
    model = model.to(device)


def train():
    model = generate_model(opt)

    loss_function = DiceBCELoss()

    # load data
    train_data = getattr(datasets, opt.dataset)(opt.root, opt.train_data_dir, mode='train')
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # train_total_batch = int(len(train_data) / 1)
    valid_data = getattr(datasets, opt.dataset)(opt.root, opt.valid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_total_batch = int(len(valid_data) / 1)

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.nEpoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')

    # Create results folder if it doesn't exist
    results_folder = '/SLG/results/SLGNET'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # Save header row to file
    filename = f"{results_folder}/{time.strftime('%Y-%m-%d_%H-%M-%S')}_epoch_results.txt"
    with open(filename, "w") as f:
        f.write("Epoch,Recall,FPR,Precision,F1,ZSI,ACC_overall,IoU_poly,IoU_bg,IoU_mean\n")

    # Initialize lists for storing metric values
    recalls = []
    FPR = []
    precisions = []
    F1s = []
    ZSIs = []
    IoU_means = []
    best_F1 = 0.0
    best_epoch = 0

    for epoch in range(opt.nEpoch):
        print('------ Epoch', epoch + 1)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)
        # total_batch = int(len(train_data))
        bar = tqdm(enumerate(train_dataloader), total=total_batch)

        for i, data in bar:
            img = data['image']
            gt = data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            output = model(img)

            # loss = BceDiceLoss()(output, gt)
            # loss = DeepSupervisionLoss(output, gt)
            loss =  loss_function(output, gt)
            # loss = FocalLoss(output, gt)
            loss.backward()

            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        scheduler.step()

        metrics_result = valid(model, valid_dataloader, val_total_batch)

        # Append metric values to lists
        recalls.append(metrics_result['recall'])
        FPR.append(metrics_result['FPR'])
        precisions.append(metrics_result['precision'])
        F1s.append(metrics_result['F1'])
        ZSIs.append(metrics_result['ZSI'])
        IoU_means.append(metrics_result['IoU_mean'])

        print("Train Result:")
        print('recall: %.3f, FPR: %.4f, precision: %.3f, F1: %.3f,'
              ' ZSI: %.3f, ACC_overall: %.3f, IoU_poly: %.3f, IoU_bg: %.3f, IoU_mean: %.3f'
              % (metrics_result['recall'], metrics_result['FPR'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['ZSI'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))

        # Update best F1 value and epoch
        if metrics_result['F1'] > best_F1:
            best_F1 = metrics_result['F1']
            best_epoch = epoch + 1

        # Save model weights
        model_weights_path = os.path.join(results_folder, 'model_weights.pth')
        torch.save(model.state_dict(), model_weights_path)
        # Save training environment
        environment_path = os.path.join(results_folder, 'training_environment.json')
        with open(environment_path, 'w') as f:
            json.dump(vars(opt), f)

        # Append epoch results to file
        with open(filename, "a") as f:
            f.write(
                f"{epoch + 1},{metrics_result['recall']},{metrics_result['FPR']},{metrics_result['precision']},{metrics_result['F1']},{metrics_result['ZSI']},{metrics_result['ACC_overall']},{metrics_result['IoU_poly']},{metrics_result['IoU_bg']},{metrics_result['IoU_mean']}\n")




    print("Best F1: %.3f (Epoch %d)" % (best_F1, best_epoch))





if __name__ == '__main__':

    if opt.mode == 'train':
        print('---Cervix Train---')
        train()

    print('Done')

