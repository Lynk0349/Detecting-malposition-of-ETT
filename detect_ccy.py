from genericpath import isdir

from scipy.fftpack import ss_diff
import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
import time
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from coco_eval import COCOGenerator
import math
import statistics
from openpyxl import Workbook
import pydicom

from skimage import morphology
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test_folder", type=str, default=None, help="where is your test folder")
parser.add_argument("-a", "--anno_file", type=str, default=None, help="where is your annotation file (.json)")
parser.add_argument("-w", "--weight", type=str, default="./checkpoint/final_120.pth", help="where is your weight (.pth)")
opt = parser.parse_args()

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

def EU_distance(x1,y1,x2,y2):
    return math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))

def Gaussian_mask(shape=(3,3),sigma=0.5, shift=None):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    if (shift != None):
        M = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
        h = cv2.warpAffine(h, M, (shape[1], shape[0]))
    return h

if __name__=="__main__":
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 40)]
    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        class_num=4
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=True

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

        #inference
        score_threshold=0.5
        nms_iou_threshold=0.5
        max_detection_boxes_num=300

    model=FCOSDetector(mode="inference",config=Config)
    model.load_state_dict(torch.load(opt.weight,map_location=torch.device('cuda')))
    model = model.cuda().eval()
    print("===>success loading model")

    # print(model)

    import os
    root=opt.test_folder
    names=os.listdir(root)
    generator=COCOGenerator(opt.test_folder, opt.anno_file, dcm=True)
    generatorGT = COCOGenerator(opt.test_folder, opt.anno_file, resize_size = None, dcm=True)
    index = 0
    pysical_size = 0.139
    tp_losses = []
    tp_losses_name = []
    bp_losses = []
    bp_losses_name = []
    ETT_carina_distance_losses = []
    ETT_carina_distance_name = []
    ETT_carina_distance_GT = []
    ETT_carina_GT_name = []
    ETT_carina_distance_pred = []

    for n in tqdm(range(len(names))):
        name = names[n]
        # record previous tp and bp
        center_tp = (0,0)
        center_tp_loss = 0.0
        center_bp = (0,0)
        center_bp_loss = 0.0
        c_tp_scores = 0
        c_bp_scores = 0
        # record previous tp and bp bbox
        c_tp_bbox = (0,0)
        c_tp_bbox_scores = 0
        c_bp_bbox = (0,0)
        c_bp_bbox_scores = 0
        
        ds = pydicom.dcmread(os.path.join(root, name))
        img_bgr = ds.pixel_array.astype(float)
        img_bgr = (np.maximum(img_bgr,0)/img_bgr.max()) * 255.0
        img_bgr = np.uint8(img_bgr) 
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        img_pad=preprocess_img(img_bgr,[800,1333])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img1=transforms.ToTensor()(img)
        img1= transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225],inplace=True)(img1)
        img1=img1.cuda()

        start_t=time.time()
        with torch.no_grad():
            out = model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        scores,classes,boxes, masks =out
    
        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()
        
        masks = masks[0].cpu().detach().numpy()*255
        for c in range(masks.shape[0]):
            masks[c,:,:] = cv2.bilateralFilter(masks[c,:,:], dst=-1, d=9, sigmaColor=150, sigmaSpace=150)
            ret, masks[c,:,:] = cv2.threshold(masks[c,:,:], 100, 255, cv2.THRESH_BINARY)
        # ---draw GT on jpg---
        _,_,_,scale,_= generator[index]
        _, gt_boxes, gt_labels, scale_gt, gt_masks = generatorGT[index]

        img_bgr_label = img_bgr.copy()
        tp_box = ((int(gt_boxes[0][0]),int(gt_boxes[0][1])), (int(gt_boxes[0][2]),int(gt_boxes[0][3])))
        tp_center = (round((int(gt_boxes[1][0]) + int(gt_boxes[1][2]))/2), round((int(gt_boxes[1][1]) + int(gt_boxes[1][3]))/2))
        bp_box = ((int(gt_boxes[2][0]),int(gt_boxes[2][1])), (int(gt_boxes[2][2]),int(gt_boxes[2][3])))
        bp_center = (round((int(gt_boxes[3][0]) + int(gt_boxes[3][2]))/2), round((int(gt_boxes[3][1]) + int(gt_boxes[3][3]))/2))
        
        ETT_carina_distance_gt = EU_distance(tp_center[0],tp_center[1], bp_center[0], bp_center[1])
        ETT_carina_distance_GT.append(ETT_carina_distance_gt)
        ETT_carina_GT_name.append(name)

        img_bgr_label = cv2.rectangle(img_bgr_label, tp_box[0], tp_box[1], (0,0,200),5)
        img_bgr_label = cv2.rectangle(img_bgr_label, bp_box[0], bp_box[1], (0,0,200),5)
        img_bgr_label = cv2.circle(img_bgr_label, tp_center, 1, (0,0,200),10)
        img_bgr_label = cv2.circle(img_bgr_label, bp_center, 1, (0,0,200),10)

        gt_contour,_ = cv2.findContours(image=gt_masks, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # ---view prediect---
        for i,box in enumerate(boxes):
            # tp bbox
            if (int(classes[i]) == 1):
                pt1=(round(box[0]/scale),round(box[1]/scale))
                pt2=(round(box[2]/scale),round(box[3]/scale))  
                if (c_tp_bbox == (0,0)):
                    c_tp_bbox = (pt1, pt2)
                    c_tp_bbox_scores = scores[i]
                else:
                    if(c_tp_bbox_scores < scores[i]):
                        c_tp_bbox = (pt1, pt2)
                        c_tp_bbox_scores = scores[i]
            # bp bbox
            elif(int(classes[i]== 3)):
                pt1=(round(box[0]/scale),round(box[1]/scale))
                pt2=(round(box[2]/scale),round(box[3]/scale))  
                if (c_bp_bbox == (0,0)):
                    c_bp_bbox = (pt1, pt2)
                    c_bp_bbox_scores = scores[i]
                else:
                    if(c_bp_bbox_scores < scores[i]):
                        c_bp_bbox = (pt1, pt2)
                        c_bp_bbox_scores = scores[i] 
            else:
                pass
        # tp gaussian
        if (c_tp_bbox == (0,0)):
            gaussian_mask_tp=Gaussian_mask((img_bgr.shape[0],img_bgr.shape[1]), sigma=(min(img_bgr.shape[1]//4, img_bgr.shape[0]//4)//2))
        else:
            x_shift=round((c_tp_bbox[0][0] + c_tp_bbox[1][0])/2 - (img_bgr.shape[1]/2))
            y_shift=round((c_tp_bbox[0][1] + c_tp_bbox[1][1])/2 - (img_bgr.shape[0]/2))
            gaussian_mask_tp=Gaussian_mask((img_bgr.shape[0],img_bgr.shape[1]), sigma=(min(img_bgr.shape[1]//4, img_bgr.shape[0]//4)//2), shift=(x_shift, y_shift))
        # bp gaussian
        if (c_bp_bbox == (0,0)):
            gaussian_mask_bp=Gaussian_mask((img_bgr.shape[0],img_bgr.shape[1]), sigma=(min(img_bgr.shape[1]//4, img_bgr.shape[0]//4)//2))
        else:
            x_shift=round((c_bp_bbox[0][0] + c_bp_bbox[1][0])/2 - (img_bgr.shape[1]/2))
            y_shift=round((c_bp_bbox[0][1] + c_bp_bbox[1][1])/2 - (img_bgr.shape[0]/2))
            gaussian_mask_bp=Gaussian_mask((img_bgr.shape[0],img_bgr.shape[1]), sigma=(min(img_bgr.shape[1]//4, img_bgr.shape[0]//4)//2), shift=(x_shift, y_shift))
        # preserve the heightest score of point
        for i,box in enumerate(boxes):
            if ((int(classes[i]) == 2) | (int(classes[i]) == 4)):
                center = (round(((int(box[0]) + int(box[2]))/2)/scale), round(((int(box[1]) + int(box[3]))/2)/scale)) # (x,y)
                if (int(classes[i]) == 2):
                    if (len(tp_losses_name) == 0):
                        tp_losses_name.append(name)
                        center_tp = center
                        c_tp_scores = scores[i]
                    else:
                        # preserve the heightest score of tp
                        if(name == tp_losses_name[-1]):
                            if (gaussian_mask_tp[center_tp[1]][center_tp[0]]*c_tp_scores >= gaussian_mask_tp[center[1]][center[0]]*scores[i]):
                                pass
                            else:
                                center_tp = center
                                c_tp_scores = scores[i]
                        else:
                            tp_losses_name.append(name)
                            center_tp = center
                            c_tp_scores = scores[i]

                else:
                    if (len(bp_losses_name) == 0):
                        bp_losses_name.append(name)
                        center_bp = center
                        c_bp_scores = scores[i]
                    else:
                        # preserve the heightest score of bp
                        if(name == bp_losses_name[-1]):
                            if (gaussian_mask_bp[center_bp[1]][center_bp[0]]*c_bp_scores >= gaussian_mask_bp[center[1]][center[0]]*scores[i]):
                                pass
                            else:
                                center_bp = center
                                c_bp_scores = scores[i]
                        else:
                            bp_losses_name.append(name)
                            center_bp = center
                            c_bp_scores = scores[i]
            else:
                pass

        # ---masks---
        masks_skel = morphology.skeletonize(masks[1,:,:]/255, method='lee')
        # ---find tip---
        masks_point_ori=(0,0)
        horizontal_pixel_sum = np.sum(masks_skel, axis=1)
        # ---find y---
        find_y = 0
        dst_y = 0        
        for y in range(len(horizontal_pixel_sum)):
            if(find_y == 0):
                if(horizontal_pixel_sum[y] > 0):
                    find_y += 1
                else:
                    pass
            elif(find_y == 1):
                if(horizontal_pixel_sum[y] == 0):
                    find_y += 1
                    dst_y = y-1
                    break
                else:
                    pass
            else:
                pass
        # ---find x
        find_x = 0
        dst_x = 0
        x1 = 0
        x2 = 0
        for x in range(len(masks_skel[dst_y])):
            if(find_x == 0):
                if(masks_skel[dst_y][x] > 0):
                    find_x += 1
                    x1 = x
                else:
                    pass
            elif(find_x == 1):
                if(masks_skel[dst_y][x] == 0):
                    find_x += 1
                    x2 = x-1
                    break
                else:
                    pass
            else:
                pass
        if (x1 != 0):
            dst_x = round((x1+x2)/2)
        masks_point_ori = (round(dst_x/scale), round(dst_y/scale))
        img_bgr = cv2.circle(img_bgr, masks_point_ori, 1, (230,230,0), 10)

        # ---draw prediect on jpg---
        if (center_tp != (0,0)):
            if (masks_point_ori != (0,0)):
                mask_and_tp_dis = EU_distance(center_tp[0], center_tp[1], masks_point_ori[0], masks_point_ori[1])
                # if (mask_and_tp_dis >= 100):
                #     center_tp = masks_point_ori
                # else:
                #     pass
            else:
                if (c_tp_bbox != (0,0)):
                    if (c_tp_bbox_scores > c_tp_scores and c_tp_scores < 0.75):
                        center_tp = (round((c_tp_bbox[0][0] + c_tp_bbox[1][0])/2), c_tp_bbox[1][1])
                    else:
                        pass
                else:
                    pass
            img_bgr = cv2.circle(img_bgr, center_tp, 1, (200,25,25),10)  
            center_tp_loss = EU_distance(center_tp[0], center_tp[1], tp_center[0], tp_center[1])*pysical_size
            tp_losses.append(center_tp_loss)
            cv2.putText(img_bgr, "%.3f mm"%(center_tp_loss), (round(center_tp[0]), round(center_tp[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,(200,25,25), thickness=3)
        
        if (center_bp != (0,0)):
            if (c_bp_bbox != (0,0)):
                if (c_bp_bbox_scores > c_bp_scores and c_bp_scores < 0.75):
                    center_bp = (round((c_bp_bbox[0][0]+c_bp_bbox[1][0])/2), round((c_bp_bbox[0][1] + c_bp_bbox[1][1])/2))
                else:
                    pass
            else:
                pass
            img_bgr = cv2.circle(img_bgr, center_bp, 1, (0,200,0),10)
            center_bp_loss = EU_distance(center_bp[0], center_bp[1], bp_center[0], bp_center[1])*pysical_size
            bp_losses.append(center_bp_loss)
            cv2.putText(img_bgr, "%.3f mm"%(center_bp_loss), (round(center_bp[0]), round(center_bp[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,200,0), thickness=3)
        
        if (c_tp_bbox != (0,0)):
            img_bgr = cv2.rectangle(img_bgr, c_tp_bbox[0], c_tp_bbox[1] ,(200,25,25),5)
        if (c_bp_bbox != (0,0)):
            img_bgr = cv2.rectangle(img_bgr, c_bp_bbox[0], c_bp_bbox[1] ,(0,200,0),5) 

        # Calculate distance loss between ETT and carina
        if ((center_tp != (0,0)) and (center_bp != (0,0))):
            ETT_carina_distance_dt = EU_distance(center_tp[0], center_tp[1], center_bp[0], center_bp[1])
            ETT_carina_distance_loss = abs(ETT_carina_distance_dt - ETT_carina_distance_gt)*pysical_size
            ETT_carina_distance_losses.append(ETT_carina_distance_loss)
            ETT_carina_distance_name.append(name)
            ETT_carina_distance_pred.append(ETT_carina_distance_dt)
        
            

        alpha = 0.7
        beta = 1-alpha
        gamma = 0
        img_pad = cv2.addWeighted(img_bgr, alpha, img_bgr_label, beta, gamma) 
        
        if not os.path.isdir('out_images'):
            os.mkdir('out_images')
        cv2.imwrite('out_images/{}'.format(name.split('.')[0]+".jpg"), img_pad)
        index+=1
        # ---

    # recall and precision
    total_nums = len(names) #=TP + FN
    tp_true_positive_num = 0
    tp_false_positive_num = 0
    
    bp_true_positive_num = 0
    bp_false_positive_num = 0
    
    tp_20 = 0
    tp_15 = 0
    tp_10 = 0
    tp_5 = 0
    bp_20 = 0
    bp_15 = 0
    bp_10 = 0
    bp_5 = 0
    ETT_carina_20 = 0
    ETT_carina_15 = 0
    ETT_carina_10 = 0
    ETT_carina_5 = 0

    # recall, precision and distribution
    for i in range(len(tp_losses)):
        # Count true poistive and false positve
        if (tp_losses[i] > 10):
            tp_false_positive_num += 1
        else:
            tp_true_positive_num += 1
        # Count distribution
        if (15 < tp_losses[i] <= 20):
            tp_20 += 1
        elif (10 < tp_losses[i] <= 15):
            tp_15 += 1
        elif (5 < tp_losses[i] <= 10):
            tp_10 += 1
        elif (0 <= tp_losses[i] <= 5):
            tp_5 += 1
        else:
            pass

    for i in range(len(bp_losses)):
        # Count true poistive and false positve
        if (bp_losses[i] > 10):
            bp_false_positive_num += 1
        else:
            bp_true_positive_num += 1
        # Count distribution
        if (15 < bp_losses[i] <= 20):
            bp_20 += 1
        elif (10 < bp_losses[i] <= 15):
            bp_15 += 1
        elif (5 < bp_losses[i] <= 10):
            bp_10 += 1
        elif (0 <= bp_losses[i] <= 5):
            bp_5 += 1
        else:
            pass
    
    for i in range(len(ETT_carina_distance_losses)):
        # Count distribution
        if (15 < ETT_carina_distance_losses[i] <= 20):
            ETT_carina_20 += 1
        elif (10 < ETT_carina_distance_losses[i] <= 15):
            ETT_carina_15 += 1
        elif (5 < ETT_carina_distance_losses[i] <= 10):
            ETT_carina_10 += 1
        elif (0 <= ETT_carina_distance_losses[i] <= 5):
            ETT_carina_5 += 1
        else:
            pass
    
    tp_recall = tp_true_positive_num/total_nums
    tp_precision = tp_true_positive_num/(tp_true_positive_num+tp_false_positive_num)
    bp_recall = bp_true_positive_num/total_nums
    bp_precision = bp_true_positive_num/(bp_true_positive_num+bp_false_positive_num)
    
    bp_avg_loss = statistics.mean(bp_losses)
    bp_st_dev = statistics.pstdev(bp_losses)
    tp_avg_loss = statistics.mean(tp_losses)
    tp_st_dev = statistics.pstdev(tp_losses)
    ETT_carina_avg_loss = statistics.mean(ETT_carina_distance_losses)
    ETT_carina_st_dev = statistics.pstdev(ETT_carina_distance_losses)
    
    # confision matrix
    GT_suitable = []
    GT_unsuitable = []
    pred_suitable = []
    pred_unsuitable = []
    pred_undetect = []
    gt_dt_ss = 0
    gt_dt_su = 0
    gt_dt_sud = 0
    gt_dt_us = 0
    gt_dt_uu = 0
    gt_dt_uud = 0

    for i in range(len(ETT_carina_distance_GT)):
        if (20 <= (ETT_carina_distance_GT[i]*pysical_size) <=70):
            GT_suitable.append(ETT_carina_GT_name[i])
        else:
            GT_unsuitable.append(ETT_carina_GT_name[i])

    for i in range(len(ETT_carina_distance_pred)):
        if (20 <= (ETT_carina_distance_pred[i]*pysical_size) <= 70):
            pred_suitable.append(ETT_carina_distance_name[i])
        else:
            pred_unsuitable.append(ETT_carina_distance_name[i])

    for i in range(len(GT_suitable)):
        if (GT_suitable[i] in pred_suitable):
            gt_dt_ss += 1
        elif (GT_suitable[i] in pred_unsuitable):
            gt_dt_su += 1
        else:
            gt_dt_sud += 1
    for i in range(len(GT_unsuitable)):
        if (GT_unsuitable[i] in pred_suitable):
            gt_dt_us += 1
        elif (GT_unsuitable[i] in pred_unsuitable):
            gt_dt_uu += 1
        else:
            gt_dt_uud += 1

    # # ---excel---
    excel_file = Workbook()
    sheet = excel_file.active
    sheet["A1"] = 'branch_point_loss'
    sheet["A2"] = "Num"
    sheet["B2"] = "Image Name"
    sheet["C2"] = "losses (mm)"
    
    sheet["E1"] = 'tube_point_loss'
    sheet["E2"] = 'Num'
    sheet["F2"] = "Image Name"
    sheet["G2"] = "losses (mm)"

    sheet["I1"] = 'ETT_carina_distance'
    sheet["I2"] = "Num"
    sheet["J2"] = "Image Name"
    sheet["K2"] = "losses (mm)"

    # recall and precision
    sheet["M2"] = 'tp'
    sheet["M3"] = 'Recall'
    sheet["M4"] = "{:.2f}%".format(tp_recall*100)
    sheet["N3"] = 'Precision'
    sheet["N4"] = "{:.2f}%".format(tp_precision*100)

    sheet["M6"] = 'bp'
    sheet["M7"] = 'Recall'
    sheet["M8"] = "{:.2f}%".format(bp_recall*100)
    sheet["N7"] = 'Precision'
    sheet["N8"] = "{:.2f}%".format(bp_precision*100)
    
    # distribution
    sheet["M10"] = 'tp'
    sheet["M11"] = '<= 5 mm'
    sheet["N11"] = '<= 10 mm'
    sheet["O11"] = '<= 15 mm'
    sheet["P11"] = '<= 20 mm'
    sheet["M12"] = "{:.2f}%".format((tp_5/total_nums)*100)
    sheet["N12"] = "{:.2f}%".format(((tp_5+tp_10)/total_nums)*100)
    sheet["O12"] = "{:.2f}%".format(((tp_5+tp_10+tp_15)/total_nums)*100)
    sheet["P12"] = "{:.2f}%".format(((tp_5+tp_10+tp_15+tp_20)/total_nums)*100)

    sheet["M14"] = 'bp'
    sheet["M15"] = '<= 5 mm'
    sheet["N15"] = '<= 10 mm'
    sheet["O15"] = '<= 15 mm'
    sheet["P15"] = '<= 20 mm'
    sheet["M16"] = "{:.2f}%".format((bp_5/total_nums)*100)
    sheet["N16"] = "{:.2f}%".format(((bp_5+bp_10)/total_nums)*100)
    sheet["O16"] = "{:.2f}%".format(((bp_5+bp_10+bp_15)/total_nums)*100)
    sheet["P16"] = "{:.2f}%".format(((bp_5+bp_10+bp_15+bp_20)/total_nums)*100)


    sheet["M18"] = 'ETT-carina'
    sheet["M19"] = '<= 5 mm'
    sheet["N19"] = '<= 10 mm'
    sheet["O19"] = '<= 15 mm'
    sheet["P19"] = '<= 20 mm'
    sheet["M20"] = "{:.2f}%".format((ETT_carina_5/total_nums)*100)
    sheet["N20"] = "{:.2f}%".format(((ETT_carina_5+ETT_carina_10)/total_nums)*100)
    sheet["O20"] = "{:.2f}%".format(((ETT_carina_5+ETT_carina_10+ETT_carina_15)/total_nums)*100)
    sheet["P20"] = "{:.2f}%".format(((ETT_carina_5+ETT_carina_10+ETT_carina_15+ETT_carina_20)/total_nums)*100)

    # confision matrix
    sheet["M22"] = 'Confusion Matrix'
    sheet["M23"] = 'Predict\GT'
    sheet["N23"] = 'Suitable'
    sheet["O23"] = 'Unsuitable'
    sheet["M24"] = 'Suitable'
    sheet["M25"] = 'Unsuitable'
    sheet["M26"] = 'Undetection'
    
    sheet["N24"] = "{}".format(gt_dt_ss)
    sheet["N25"] = "{}".format(gt_dt_su)
    sheet["N26"] = "{}".format(gt_dt_sud)
    sheet["O24"] = "{}".format(gt_dt_us)
    sheet["O25"] = "{}".format(gt_dt_uu)
    sheet["O26"] = "{}".format(gt_dt_uud)


    for i, loss in enumerate(bp_losses):
        sheet["A"+str(i+3)] = str(i+1)
        sheet["B"+str(i+3)] = bp_losses_name[i]
        sheet["C"+str(i+3)] = loss
    sheet["A"+str(i+3+1)] = "Average loss"
    sheet["C"+str(i+3+1)] = "%.3f mm"%bp_avg_loss
    sheet["A"+str(i+3+1+1)] = "Standard deviation"
    sheet["C"+str(i+3+1+1)] = "%.3f mm"%bp_st_dev

    for j, loss in enumerate(tp_losses):
        sheet["E"+str(j+3)] = str(j+1)
        sheet["F"+str(j+3)] = tp_losses_name[j]
        sheet["G"+str(j+3)] = loss
    sheet["E"+str(j+3+1)] = "Average loss"
    sheet["G"+str(j+3+1)] = "%.3f mm"%tp_avg_loss
    sheet["E"+str(j+3+1+1)] = "Standard deviation"
    sheet["G"+str(j+3+1+1)] = "%.3f mm"%tp_st_dev

    for k, loss in enumerate(ETT_carina_distance_losses):
        sheet["I"+str(k+3)] = str(k+1)
        sheet["J"+str(k+3)] = ETT_carina_distance_name[k]
        sheet["K"+str(k+3)] = loss
    sheet["I"+str(j+3+1)] = "Average loss"
    sheet["K"+str(j+3+1)] = "%.3f mm"%ETT_carina_avg_loss
    sheet["I"+str(j+3+1+1)] = "Standard deviation"
    sheet["K"+str(j+3+1+1)] = "%.3f mm"%ETT_carina_st_dev

    excel_file.save("losses_ccy.xlsx")
    # # ---


