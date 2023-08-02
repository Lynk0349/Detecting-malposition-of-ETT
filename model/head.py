import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg
        
        cls_branch=[]
        reg_branch=[]

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)

        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits=nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=3,padding=1)
        
        self.apply(self.init_conv_RandomNormal)
        
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):
            cls_conv_out=self.cls_conv(P)
            reg_conv_out=self.reg_conv(P)
            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits,cnt_logits,reg_preds

class SegHead_w_fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SegHead_w_fusion, self).__init__()

        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.GroupNorm(32,in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.GroupNorm(32,in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.GroupNorm(32,in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.GroupNorm(32,in_channel),
            nn.ReLU(inplace=True),
        )

        self.seg_pred = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')

    def forward(self, inputs, cat):
        seg_preds=[]
        P3, P4, P5 = inputs

        C2 = cat

        P2 = C2 + self.upsamplelike([P3,C2]) + self.upsamplelike([P4,C2])+ self.upsamplelike([P5,C2])
        seg_conv_out = self.seg_conv(P2)
        seg_conv_out = self.seg_pred(F.interpolate(P2, size=(P2.shape[2]*4, P2.shape[3]*4), mode='nearest'))
        seg_preds.append(seg_conv_out)

        return seg_preds

