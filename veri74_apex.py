from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import numpy.ma as ma
import warnings
import pickle
from skimage import io, transform
from PIL import Image
import os.path as osp
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, utils

import models
from losses import triplet_loss, xent_loss
from loss import CrossEntropyLoss, TripletLoss
from utils.avgmeter import AverageMeter
from utils.iotools import check_isfile
from utils.loggers import Logger, RankLogger
from utils.torchtools import count_num_param, accuracy, load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from utils.visualtools import visualize_ranked_results
from utils.generaltools import set_random_seed
from optimizers import init_optimizer
from lr_schedulers import init_lr_scheduler
from functions import keyfromval, strint, ranges, search
from datasets import init_imgreid_dataset
from data_loading import VeriDataset as vd
#from transform import Rescale, RandomCrop, ToTensor 
from transform import train_transforms
from test_loading import ImageDataManager
from evaluation import evaluate
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from utils.reranking import re_ranking,re_ranking_numpy


def test(model, queryloader, galleryloader, batch_size, use_gpu, ranks=[1, 5, 10], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    #distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 10)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')
    
    return cmc[0], distmat
def test_rerank(model, queryloader, galleryloader, batch_size, use_gpu, ranks=[1, 5, 10], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, batch_size))

    m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.numpy()

    distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 10)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')
    
    return cmc[0], distmat

def main():
    #GENERAL
    root = "/home/kuru/Desktop/veri-gms-master/"
    train_dir = '/home/kuru/Desktop/veri-gms-master/VeRispan/image_train/'
    source = {'veri'}
    target = {'veri'}
    workers = 2
    height = 320
    width  = 320
    train_sampler = 'RandomSampler'

    #AUGMENTATION
    random_erase = True
    jitter = True
    aug = True

    #OPTIMIZATION
    opt = 'adam'
    lr = 0.0003
    weight_decay = 5e-4
    momentum = 0.9
    sgd_damp = 0.0
    nesterov = True
    warmup_factor = 0.01
    warmup_method = 'linear'

    #HYPERPARAMETER
    max_epoch = 80
    start = 0
    train_batch_size = 16
    test_batch_size = 50

    #SCHEDULER
    lr_scheduler = 'multi_step'
    stepsize = [30, 60]
    gamma = 0.1

    #LOSS
    margin = 0.3
    num_instances = 6
    lambda_tri = 1

    #MODEL
    arch = 'resnet101_ibn_a'
    no_pretrained = False

    #TEST SETTINGS
    load_weights = '/home/kuru/Desktop/veri-gms-master/IBN-Net_pytorch0.4.1/resnet101_ibn_a.pth'
    #load_weights = None
    start_eval = 0
    eval_freq = -1

    #MISC
    use_gpu = True
    use_amp =True
    print_freq = 50
    seed = 1
    resume = ''
    save_dir = '/home/kuru/Desktop/veri-gms-master/logapex/'
    gpu_id = 0
    vis_rank = True
    query_remove = True
    evaluate = False

    dataset_kwargs = {
        'source_names': source,
        'target_names': target,
        'root': root,
        'height': height,
        'width': width,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'train_sampler': train_sampler,
        'random_erase': random_erase,
        'color_jitter': jitter,
        'color_aug': aug
        }
    transform_kwargs = {
        'height': height,
        'width': width,
        'random_erase': random_erase,
        'color_jitter': jitter,
        'color_aug': aug
    }

    optimizer_kwargs = {
        'optim': opt,
        'lr': lr,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'sgd_dampening': sgd_damp,
        'sgd_nesterov': nesterov
        }

    lr_scheduler_kwargs = {
        'lr_scheduler': lr_scheduler,
        'stepsize': stepsize,
        'gamma': gamma
        }
    
    use_gpu = torch.cuda.is_available()
    log_name = 'log_test.txt' if evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(save_dir, log_name))
    print('Currently using GPU ', gpu_id)
    cudnn.benchmark = True

    print('Initializing image data manager')
    dataset = init_imgreid_dataset(root='/home/kuru/Desktop/veri-gms-master/', name='veri')
    train = []
    num_train_pids = 0
    num_train_cams = 0


    for img_path, pid, camid in dataset.train:
        path = img_path[52:77]
        #print(path)
        folder = path[1:4]
        pid += num_train_pids
        camid += num_train_cams
        train.append((path, folder, pid, camid))

    num_train_pids += dataset.num_train_pids
    num_train_cams += dataset.num_train_cams

    pid = 0
    pidx = {}
    for img_path, pid, camid in dataset.train:
        path = img_path[52:77]
        
        folder = path[1:4]
        pidx[folder] = pid
        pid+= 1

    
    path = '/home/kuru/Desktop/veri-gms-master/gms/'
    pkl = {}
    entries = os.listdir(path)
    for name in entries:
        f = open((path+name), 'rb')
        if name=='featureMatrix.pkl':
            s = name[0:13]
        else:
            s = name[0:3]
        pkl[s] = pickle.load(f)
        f.close


    
    transform_t = train_transforms(**transform_kwargs)

    data_tfr = vd(pkl_file='index.pkl', dataset = train, root_dir='/home/kuru/Desktop/veri-gms-master/VeRi/image_train/', transform=transform_t)
    trainloader = DataLoader(data_tfr, sampler=None,batch_size=train_batch_size, shuffle=True, num_workers=workers,pin_memory=False, drop_last=True)

    #data_tfr = vd(pkl_file='index.pkl', dataset = train, root_dir=train_dir,transform=transforms.Compose([Rescale(64),RandomCrop(32),ToTensor()]))
    #dataloader = DataLoader(data_tfr, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print('Initializing test data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs)
    testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(arch))
    model = models.init_model(name=arch, num_classes=num_train_pids, loss={'xent', 'htri'}, last_stride = 1,
                              pretrained=not no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if load_weights is not None:
        print("weights loaded")
        load_pretrained_weights(model, load_weights)

    model = (model).cuda() if use_gpu else model

    #model = nn.DataParallel(model).cuda() if use_gpu else model
    optimizer = init_optimizer(model, **optimizer_kwargs)
    #optimizer = init_optimizer(model)
    
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2", 
        keep_batchnorm_fp32=True, loss_scale="dynamic"
        )
    model = nn.DataParallel(model).cuda() if use_gpu else model
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs)

    criterion_xent = CrossEntropyLoss(num_classes=num_train_pids, use_gpu=use_gpu, label_smooth=True)
    criterion_htri = TripletLoss(margin=margin)
    ranking_loss = nn.MarginRankingLoss(margin = margin)

    if evaluate:
        print('Evaluate only')

        for name in target:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            _, distmat = test(model, queryloader, galleryloader, train_batch_size, use_gpu, return_distmat=True)

            if vis_rank:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(save_dir, 'ranked_results', name),
                    topk=20
                )
        return    

    time_start = time.time()
    ranklogger = RankLogger(source, target)
    print('=> Start training')

    data_index = search(pkl)
    
    for epoch in range(start, max_epoch):
        losses = AverageMeter()
        #xent_losses = AverageMeter()
        htri_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()

        model.train()
        for p in model.parameters():
            p.requires_grad = True    # open all layers

        end = time.time()
        for batch_idx, (img,label,index,pid, cid) in enumerate(trainloader):
            trainX, trainY = torch.zeros((train_batch_size*3,3,height, width), dtype=torch.float32), torch.zeros((train_batch_size*3), dtype = torch.int64)
            #pids = torch.zeros((batch_size*3), dtype = torch.int16)
            for i in range(train_batch_size):
 
                labelx = str(label[i])
                indexx = int(index[i])
                cidx = int(pid[i])
                if indexx >len(pkl[labelx])-1:
                    indexx = len(pkl[labelx])-1

                #maxx = np.argmax(pkl[labelx][indexx])
                a = pkl[labelx][indexx]
                minpos = np.argmax(ma.masked_where(a==0, a)) 
                pos_dic = data_tfr[data_index[cidx][1]+minpos]

                neg_label = int(labelx)
                while True:
                    neg_label = random.choice(range(1, 770))
                    if neg_label is not int(labelx) and os.path.isdir(os.path.join('/home/kuru/Desktop/adiusb/veri-split/train', strint(neg_label))) is True:
                        break
                negative_label = strint(neg_label)
                neg_cid = pidx[negative_label]
                neg_index = random.choice(range(0, len(pkl[negative_label])))

                neg_dic = data_tfr[data_index[neg_cid][1]+neg_index]
                trainX[i] = img[i]
                trainX[i+train_batch_size] = pos_dic[0]
                trainX[i+(train_batch_size*2)] = neg_dic[0]
                trainY[i] = cidx
                trainY[i+train_batch_size] = pos_dic[3]
                trainY[i+(train_batch_size*2)] = neg_dic[3]

                #print("anc",labelx,'posdic', pos_dic[1],pos_dic[2],'negdic', neg_dic[1],neg_dic[2])

        
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            outputs, features = model(trainX)
            xent_loss = criterion_xent(outputs[0:train_batch_size], trainY[0:train_batch_size])
            htri_loss = criterion_htri(features, trainY)

            #tri_loss = ranking_loss(features)
            #ent_loss = xent_loss(outputs[0:batch_size], trainY[0:batch_size], num_train_pids)
            
            loss = htri_loss+xent_loss
            optimizer.zero_grad()


            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        
            #loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            losses.update(loss.item(), trainY.size(0))
            htri_losses.update(htri_loss.item(), trainY.size(0))
            accs.update(accuracy(outputs[0:train_batch_size], trainY[0:train_batch_size])[0])

            if (batch_idx) % print_freq == 0:
                print('Train ', end=" ")
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    loss = htri_losses,
                    acc=accs
                ))

            end = time.time()
        
        scheduler.step()
        print('=> Test')

        for name in target:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            rank1, distmat = test(model, queryloader, galleryloader, test_batch_size, use_gpu)
            ranklogger.write(name, epoch + 1, rank1)
            rank2, distmat2 = test_rerank(model, queryloader, galleryloader, test_batch_size, use_gpu)
            ranklogger.write(name, epoch + 1, rank2)
        del queryloader
        del galleryloader
        del distmat
        #print(torch.cuda.memory_allocated(),torch.cuda.memory_cached())
        torch.cuda.empty_cache()


        if (epoch + 1) == max_epoch:
        #if (epoch + 1) % 10 == 0:
            print('=> Test')
            save_checkpoint({
                'state_dict': model.state_dict(),
                'rank1': rank1,
                'epoch': epoch + 1,
                'arch': arch,
                'optimizer': optimizer.state_dict(),
            }, save_dir)



            if vis_rank:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(save_dir, 'ranked_results', name),
                    topk=20)
        

if __name__ == '__main__':
    main()