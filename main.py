

import cv2
import numpy as np
import json
import math
import os
import argparse
import time
import copy

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import os
import torch
import numpy as np
import sys

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader, ImageDataLoader_Val_Test
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
parser.add_argument('--DATA_ROOT', help='Location to root directory for dataset reading') # Data/fdst
parser.add_argument('--SAVE_ROOT', help='Location to root directory for saving checkpoint models') # Output
parser.add_argument('--Dataset', default='fdst', 
                    type=str, help='Dataset types are fdst, mall, and ucsd')

parser.add_argument('--MODE', default='train',
                    help='MODE can be all, train, val, and test')

parser.add_argument('--MAX_EPOCHS', default=1, 
                    type=int, help='Number of training epoc')
parser.add_argument('--VAL_EPOCHS', default=1, 
                    type=int, help='Number of training epoc')
parser.add_argument('-l','--learning_rate', 
                    default=0.001, type=float, help='initial learning rate')

## Parse arguments
args = parser.parse_args()




def train(net, train_loader,optimizer, num_epochs):
    log_file = open(args.SAVE_ROOT+"/"+args.Dataset+"_training.log","w",1)
    log_print("Training ....", color='green', attrs=['bold'])
    # training
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()
    for epoch in range(1,num_epochs+1):
        step = -1
        train_loss = 0
        for blob in train_loader:                
            step = step + 1        
            im_data = blob['data']
            gt_data = blob['gt_density']
            density_map = net(im_data, gt_data)
            loss = net.loss
            train_loss += loss.data
            step_cnt += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % disp_interval == 0:            
                duration = t.toc(average=False)
                fps = step_cnt / duration
                gt_count = np.sum(gt_data)    
                density_map = density_map.data.cpu().numpy()
                et_count = np.sum(density_map)
                utils.save_results(im_data,gt_data,density_map, args.SAVE_ROOT)
                log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                    step, 1./fps, gt_count,et_count)
                log_print(log_text, color='green', attrs=['bold'])
                re_cnt = True   
            if re_cnt:                                
                t.tic()
                re_cnt = False
    return net

def val(net,val_path,optimizer, num_epochs, Dataset=args.Dataset):

    if Dataset=="fdst":
      num_sessions=3
      val_len=300
      low_limit=1
      high_limit=300
    else:
        num_sessions=8
        val_len=1200
        low_limit=401
        high_limit=1200
    #print(num_sessions)


    sessions_list = []
    ses_size = 100
    
    for i in range(low_limit, high_limit,ses_size): 
      sessions_list.append(i)
    sessions_list.append(val_len)
    #print("Validation list: ", sessions_list)
    for val_inc in range(len(sessions_list)-1):
        start_frame = sessions_list[val_inc]
        end_frame = sessions_list[val_inc+1]
        #print('start:,end:', (start_frame,end_frame))

        val_loader = ImageDataLoader_Val_Test(val_path, None,'validation_split',start_frame, end_frame, shuffle=False, gt_downsample=True, pre_load=True, Dataset="ucsd")
        log_file = open(args.SAVE_ROOT+"/"+args.Dataset+"_validation.log","w",1)
        log_print("Validation/Self Training ....", color='green', attrs=['bold'])
        # training
        train_loss = 0
        step_cnt = 0
        re_cnt = False
        t = Timer()
        t.tic()
        for epoch in range(1,num_epochs+1):
            step = -1
            train_loss = 0
            for blob in val_loader:                
                step = step + 1        
                im_data = blob['data']
                net.training = False
                gt_data = net(im_data)
                gt_data = gt_data.cpu().detach().numpy()
                net.training = True
                density_map = net(im_data, gt_data)
                loss = net.loss
                train_loss += loss.data
                step_cnt += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if step % disp_interval == 0:            
                  duration = t.toc(average=False)
                  fps = step_cnt / duration
                  gt_count = np.sum(gt_data)    
                  density_map = density_map.data.cpu().numpy()
                  et_count = np.sum(density_map)
                  utils.save_results(im_data,gt_data,density_map, args.SAVE_ROOT)
                  log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                      step, 1./fps, gt_count,et_count)
                  log_print(log_text, color='green', attrs=['bold'])
                  re_cnt = True   
                if re_cnt:                                
                  t.tic()
                  re_cnt = False

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False 

        session= str(sessions_list[val_inc])
        network.save_net(args.SAVE_ROOT+'/'+args.Dataset+ session +'_Self_trained_model.h5', net) 
        output_dir = './densitymaps/' + session 
        net.cuda()
        net.eval()

        all_val_loader = ImageDataLoader(val_path, None, 'validation_split', shuffle=False, gt_downsample=True, pre_load=True , Dataset=args.Dataset)

        for blob in all_val_loader:                        
            im_data = blob['data']
            net.training = False
            density_map = net(im_data)
            density_map = density_map.data.cpu().numpy()
            new_dm= density_map.reshape([ density_map.shape[2], density_map.shape[3] ])
            
            np.savetxt(output_dir + '_output_' + blob['fname'].split('.')[0] +'.csv', new_dm, delimiter=',', fmt='%.6f')

    return net

def test(net,test_path,optimizer, num_epochs, Dataset=args.Dataset):

    if Dataset=="fdst":
      num_sessions=3
      test_len=300
      low_limit=1
      high_limit=300
    else:
        num_sessions=8
        test_len=1200
        low_limit=401
        high_limit=1200
    #print(num_sessions)


    sessions_list = []
    ses_size = 100
    
    for i in range(low_limit, high_limit,ses_size): 
      sessions_list.append(i)
    sessions_list.append(test_len)
    #print("test list: ", sessions_list)
    for test_inc in range(len(sessions_list)-1):
        start_frame = sessions_list[test_inc]
        end_frame = sessions_list[test_inc+1]
        #print('start:,end:', (start_frame,end_frame))

        test_loader = ImageDataLoader_Val_Test(test_path, None,'test_split',start_frame, end_frame, shuffle=False, gt_downsample=True, pre_load=True, Dataset="ucsd")
        log_file = open(args.SAVE_ROOT+"/"+args.Dataset+"_test.log","w",1)
        log_print("test/Self Training ....", color='green', attrs=['bold'])
        # training
        train_loss = 0
        step_cnt = 0
        re_cnt = False
        t = Timer()
        t.tic()
        for epoch in range(1,num_epochs+1):
            step = -1
            train_loss = 0
            for blob in test_loader:                
                step = step + 1        
                im_data = blob['data']
                net.training = False
                gt_data = net(im_data)
                gt_data = gt_data.cpu().detach().numpy()
                net.training = True
                density_map = net(im_data, gt_data)
                loss = net.loss
                train_loss += loss.data
                step_cnt += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if step % disp_interval == 0:            
                  duration = t.toc(average=False)
                  fps = step_cnt / duration
                  gt_count = np.sum(gt_data)    
                  density_map = density_map.data.cpu().numpy()
                  et_count = np.sum(density_map)
                  utils.save_results(im_data,gt_data,density_map, args.SAVE_ROOT)
                  log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                      step, 1./fps, gt_count,et_count)
                  log_print(log_text, color='green', attrs=['bold'])
                  re_cnt = True   
                if re_cnt:                                
                  t.tic()
                  re_cnt = False

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False 

        session= str(sessions_list[val_inc])
        network.save_net(args.SAVE_ROOT+'/'+args.Dataset+ session +'_Self_trained_model.h5', net) 
        output_dir = './densitymaps/' + session 
        net.cuda()
        net.eval()

        all_test_loader = ImageDataLoader(test_path, None, 'test_split', shuffle=False, gt_downsample=True, pre_load=True , Dataset=args.Dataset)

        for blob in all_test_loader:                        
            im_data = blob['data']
            net.training = False
            density_map = net(im_data)
            density_map = density_map.data.cpu().numpy()
            new_dm= density_map.reshape([ density_map.shape[2], density_map.shape[3] ])
            
            np.savetxt(output_dir + '_output_' + blob['fname'].split('.')[0] +'.csv', new_dm, delimiter=',', fmt='%.6f')

    return net



# evaluation for supervised trained model (validation)
def eval_val(net, val_path):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False 

    output_dir = './output'
    model_path = args.Dataset+ '_trained_model.h5'
    model_name = os.path.basename(model_path).split('.')[0]

    if not os.path.exists(output_dir):
           os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, 'dm_' + model_name)
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)


    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()

    val_loader = ImageDataLoader(val_path, None, 'validation_split', shuffle=False, gt_downsample=True, pre_load=True , Dataset=args.Dataset)

    for blob in val_loader:                        
        im_data = blob['data']
        net.training = False
        density_map = net(im_data)
        density_map = density_map.data.cpu().numpy()
        new_dm= density_map.reshape([ density_map.shape[2], density_map.shape[3] ])
        np.savetxt( output_dir + 'output_' + blob['fname'].split('.')[0] +'.csv', new_dm, delimiter=',', fmt='%.6f')
   
    return net


train_path = args.DATA_ROOT+'/train/input/'
train_gt_path = args.DATA_ROOT+'/train/gt/'
val_path = args.DATA_ROOT+'/val/input/'
#test_path = argparse.DATA_ROOT+'/test/input/'

#data_loader_test = ImageDataLoader(test_path, None,'test_split', shuffle=False, gt_downsample=True, pre_load=True)



#training configuration
disp_interval = 500


# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


# load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01)

pretrained_model= 'pretrained.h5'
network.load_net(pretrained_model, net)


net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)

if not os.path.exists(args.SAVE_ROOT):
    os.mkdir(args.SAVE_ROOT)

# start training
# train, validation/self training and testing model

if args.MODE == 'all' or args.MODE == 'train':
    data_loader_train = ImageDataLoader(train_path, train_gt_path,'train_split', shuffle=False, gt_downsample=True, pre_load=True, Dataset=args.Dataset)
    net = train(net, data_loader_train,optimizer,args.MAX_EPOCHS)
    network.save_net(args.SAVE_ROOT+'/'+args.Dataset+'_trained_model.h5', net) 


if args.MODE == 'all' or args.MODE == 'val':
    net = val(net,val_path, optimizer,args.VAL_EPOCHS, Dataset=args.Dataset)
    #network.save_net(args.SAVE_ROOT+'/'+args.Dataset+'_Self_trained_model.h5', net) 
    
    
# if args.MODE == 'all' or args.MODE == 'test':
#     net = test(net, test_path)




    
    




