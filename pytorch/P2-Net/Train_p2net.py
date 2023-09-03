# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
import torch
import SimpleITK as sitk
from models import unet3d
from utils.dataloader import DatasetFromFolder3D
from utils.loss import cox_regression_loss,MSE
import os
import pandas as pd
from lifelines.utils import concordance_index
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Memory(object):
    def __init__(self, lenth):
        self.lenth = lenth
        self.pred = []
        self.target = []
        self.time=[]

    def push(self, obj_pred, obj_target,times):
        self.pred.append(obj_pred)
        self.target.append(obj_target)
        self.time.append(times)

    def pop(self):
        self.pred.pop(0)
        self.target.pop(0)
        self.time.pop(0)

    def get(self):
        return self.pred, self.target,self.time

    def init_mem(self, obj_pred, obj_target,times):
        self.push(obj_pred, obj_target,times)
        if len(self.pred) < self.lenth:
            return True
        else:
            return False

def train_epoch(net, opt, loss_COX, loss_MPA, dataloader, epoch, n_epochs, Iters, memory):
    loss_log = AverageMeter()
    net.train()

    for i in range(Iters):
        net.zero_grad()
        image, target_event, target_MPA, times = next(dataloader.__iter__())

        if torch.cuda.is_available():
            image = image.cuda()
            target_event = target_event.cuda()
            target_MPA = target_MPA.cuda()
            times = times.cuda()

        pred_MPA, pred_bio = net(image)

        # Get memory
        mem_pred_bio, mem_target_event, mem_times = memory.get()

        mem_pred_bio = np.concatenate(mem_pred_bio, axis=0)
        mem_target_event = np.concatenate(mem_target_event, axis=0)
        mem_times = np.concatenate(mem_times,axis=0)

        mem_pred_bio = torch.from_numpy(mem_pred_bio)
        mem_target_event = torch.from_numpy(mem_target_event)
        mem_times = torch.from_numpy(mem_times)

        if torch.cuda.is_available():
            mem_pred_bio = mem_pred_bio.cuda()
            mem_target_event = mem_target_event.cuda()
            mem_times = mem_times.cuda()

        # Calculate loss
        pred_bio_cat = torch.cat([pred_bio, mem_pred_bio], dim=0)
        target_event_cat = torch.cat([target_event, mem_target_event], dim=0)
        times_cat = torch.cat([times, mem_times], dim=0)
        errCOX = loss_COX(pred_bio_cat, target_event_cat, times_cat)
        errMPA = loss_MPA(target_MPA, pred_MPA)
        err = errCOX + 4*errMPA
        err.backward()
        opt.step()
        loss_log.update(err.data, target_event.size(0))
        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (i + 1, Iters),
                         'train_loss %f' % (loss_log.avg)])
        print(res)
        # 更新memory
        pred_bio = pred_bio.data.cpu().numpy()
        target_event = target_event.data.cpu().numpy()
        times = times.data.cpu().numpy()
        memory.pop()
        memory.push(pred_bio, target_event, times)
    return

def model_re_weight(model_A, model_B):
    """
    Momentum update of the key encoder
    """
    for param_B, param_A in zip(model_B.parameters(), model_A.parameters()):
        param_A.data = param_B.data
    return model_A

def train_net(n_epochs=200, batch_size=1, lr=1e-4, Iters=200, model_name="P2Net-unet-enc"):
    checkpoint_dir = 'weight'
    test_dir = ''
    train_dir = ''

    pred = []
    CT = []
    event = []
    time = []

    max_cindex = 0.
    dataset_train_file = 'train.csv'
    dataset_val_file = 'val.csv'

    train_dataset = DatasetFromFolder3D(dataset_train_file, train_dir)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    net = unet3d.backbone()

    if torch.cuda.is_available():
        net = net.cuda()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_COX = cox_regression_loss()
    loss_MPA = MSE()

    # Get memory
    memory = Memory(lenth=50)
    init = True
    with torch.no_grad():
        while init:
            image, target_event, target_MPA, times = next(dataloader.__iter__())

            image = image.type(torch.FloatTensor)
            target_event = target_event.type(torch.FloatTensor)
            if torch.cuda.is_available():
                image = image.cuda()

            pred_MPA, pred_bio = net(image)

            pred_bio = pred_bio.data.cpu().numpy()
            target_event = target_event.data.cpu().numpy()
            init = memory.init_mem(pred_bio, target_event, times)

    # train
    for epoch in range(n_epochs):
        train_epoch(net, opt, loss_COX, loss_MPA, dataloader, epoch, n_epochs, Iters, memory)


        max_cindex = predict(model_name, checkpoint_dir, net, dataset_val_file, test_dir, CT, pred, event, time, max_cindex)
    torch.save(net.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, str(i) + "_" + model_name, n_epochs))



def get_cindex(pred_df):
    T = pred_df.actual_months
    E = pred_df.is_dead
    c_index = concordance_index(T, -pred_df.hazard, E)
    return c_index


def predict(model_name, checkpoint_dir, model, dataset_csv_file, img_path, CT, pred, event, time, max_cindex):
    # checkpoint_dir = 'p5-new'
    print("Predict test data")
    dataset = pd.read_csv(dataset_csv_file)
    value = {"CT": dataset['CT'].values, "event": dataset['event'].values, "time": dataset['time'].values, "MPA":dataset['MPA'].values}
    df = pd.DataFrame(value)
    model.eval()
    test_predict = []
    test_CT = []
    test_event = []
    test_time = []


    for indexs in df['CT'].index:
        # print(df['CT'].loc[indexs])
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path + 'cutImage/' + str(df['CT'].loc[indexs]) + '.mhd'))
        label = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path + 'cutLabel/' + str(df['CT'].loc[indexs]) + '_label.nii'))

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        label = np.where(label > 0, 1, 0)
        image = image * label
        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image / 2048.
        image = image[np.newaxis, np.newaxis, :, :, :]

        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():

            MPA, predict = model(image)
            predict = predict.data.cpu().numpy()

        predict = predict[0, 0]


        events = df['event'].loc[indexs]
        # print(events)
        times = df['time'].loc[indexs]
        # print(times)

        CT.append(df['CT'].loc[indexs])
        test_CT.append(df['CT'].loc[indexs])
        pred.append(predict)
        test_predict.append(predict)
        event.append(events)
        test_event.append(events)
        time.append(times)
        test_time.append(times)

    newpred = {"CT": test_CT, "predict": test_predict, "event": test_event, "time": test_time}
    pred_df = pd.DataFrame(newpred)
    pred_df = pred_df.assign(hazard=pred_df['predict'])
    pred_df = pred_df.assign(actual_months=pred_df['time'])
    pred_df = pred_df.assign(is_dead=pred_df['event'])
    c_index= get_cindex(pred_df)
    if c_index > max_cindex :
        max_cindex = c_index
        torch.save(model.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, str(c_index)))
    return max_cindex

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii", ".mhd"])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_net()