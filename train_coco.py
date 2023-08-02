import os
import math
import time
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from model.fcos import FCOSDetector
from dataset.COCO_dataset import COCODataset
from dataset.augment import Transforms


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=24, help="number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("-nc", "--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("-ng",  "--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
parser.add_argument("-d", "--data", type=str, default=None, help="where is your image training folder")
parser.add_argument("-a", "--anno", type=str, default=None, help="where is your annotation file (.json)")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def lr_func_epoch(epoch):
    lr = LR_INIT
    if epoch < WARMUP_EPOCHS:
        alpha = float(epoch) / WARMUP_EPOCHS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        if epoch < DECAY_EPOCH:
            beta = 0
            global_step = min(epoch, DECAY_EPOCH)
            cosine_decay = 0.5 * (1+math.cos(math.pi*global_step/DECAY_EPOCH))
            decayed = (1- beta) * cosine_decay + beta
            lr = lr * decayed
        else:
            lr *= 0.1

        if lr <= MINLR:
            lr = MINLR
    return float(lr)


if __name__ == '__main__':
    transform = Transforms()
    train_dataset=COCODataset(opt.data,
                            opt.anno,transform=transform, dcm=True)


    model=FCOSDetector(mode="training").cuda()
    BATCH_SIZE=opt.batch_size
    EPOCHS=opt.epochs
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn,
                                            num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))
    steps_per_epoch=len(train_dataset)//BATCH_SIZE
    TOTAL_STEPS=steps_per_epoch*EPOCHS
    WARMUP_STEPS=500
    WARMUP_FACTOR = 1.0 / 3.0
    GLOBAL_STEPS=0
    LR_INIT=1e-4
    WARMUP_EPOCHS=5
    DECAY_EPOCH=120
    MINLR = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr = LR_INIT)

    model.train()

    writer = SummaryWriter('./runs')

    for epoch in range(EPOCHS):
        tloss = 0.0
        for epoch_step,data in enumerate(train_loader):
            batch_imgs,batch_boxes,batch_classes, batch_masks = data
            batch_imgs=batch_imgs.cuda()
            batch_boxes=batch_boxes.cuda()
            batch_classes=batch_classes.cuda()
            batch_masks = batch_masks.cuda()

            lr = lr_func_epoch(epoch)
            for param in optimizer.param_groups:
                param['lr']=lr
            
            start_time=time.time()

            optimizer.zero_grad()
            losses, masks_loss = model([batch_imgs,batch_boxes,batch_classes, batch_masks])
            loss=losses[-1]+masks_loss
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3)
            optimizer.step()
            

            end_time=time.time()
            cost_time=int((end_time-start_time)*1000)
            
            GLOBAL_STEPS+=1
        
            tloss += loss.mean().item()

        avg_loss = tloss / steps_per_epoch
        if (epoch+1 == 1):
            best_loss = avg_loss
        else: 
            if(avg_loss < best_loss):
                best_loss = avg_loss
                if (epoch+1 > 80):
                    torch.save(model.state_dict(),"./checkpoint/best.pth")

        print("global_steps:%d epoch:%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f seg_loss:%.4f avg_loss:%.4f" %\
            (GLOBAL_STEPS, epoch+1, losses[0].mean().item(),losses[1].mean().item(),losses[2].mean().item(), masks_loss.mean().item(), avg_loss))
        
        writer.add_scalar("Training_Loss", avg_loss, epoch+1)
        writer.add_scalar("Learning_Rate", lr, epoch+1)
        
    torch.save(model.state_dict(),"./checkpoint/final_{}.pth".format(epoch+1))
    writer.close() 






