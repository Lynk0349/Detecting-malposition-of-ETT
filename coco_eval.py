from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms
import cv2
from model.fcos import FCOSDetector
import torch
from PIL import  Image
import pydicom
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test_folder", type=str, default=None, help="where is your test folder")
parser.add_argument("-a", "--anno_file", type=str, default=None, help="where is your annotation file (.json)")
parser.add_argument("-w", "--weight", type=str, default=None, help="where is your weight (.pth)")
opt = parser.parse_args()

class COCOGenerator(CocoDetection):
    
    def __init__(self,imgs_path,anno_path,resize_size=[800,1333], dcm=False):
        super().__init__(imgs_path,anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids=[]
        for id in self.ids:
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)
            ann=self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids=ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.resize_size=resize_size
        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]
        
        self.imgs_path = imgs_path
        self.dcm = dcm

    def __getitem__(self,index):
        
        img,ann=super().__getitem__(index)

        ann_bbox = [o for o in ann if len(o['bbox']) >= 1]
        ann_mask = [o for o in ann if len(o['segmentation']) >= 1]

        boxes = [o['bbox'] for o in ann_bbox]
        boxes=np.array(boxes,dtype=np.float32)
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]
        # mask
        masks = Image.fromarray(np.max(np.stack([self.coco.annToMask(o) * o["category_id"] for o in ann_mask]), axis=0))
        # 
        img=np.array(img)
        masks=np.array(masks)

        scale = 1
        if (self.resize_size != None):
            img,boxes,scale, masks=self.preprocess_img_boxes(img,boxes,self.resize_size, masks)
        # img=draw_bboxes(img,boxes)
        

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]
        


        img=transforms.ToTensor()(img)
        img= transforms.Normalize(self.mean, self.std,inplace=True)(img) 

        classes=np.array(classes,dtype=np.int64)

        return img,boxes,classes,scale, masks #Imageformat

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        if self.dcm == False:
            return Image.open(os.path.join(self.root, path)).convert("RGB")
        else:
            ds = pydicom.dcmread(os.path.join(self.imgs_path, path))
            img = ds.pixel_array.astype(float)
            img = (np.maximum(img,0)/img.max()) * 255.0
            img = np.uint8(img)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            return img

    def preprocess_img_boxes(self,image,boxes,input_ksize, masks):
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
        # 
        masks_resized = cv2.resize(masks, (nw,nh),interpolation = cv2.INTER_NEAREST)
        # 
        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        # 
        masks_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 1],dtype=np.uint8)
        masks_paded[:nh, :nw, 0] = masks_resized
        # 
        if boxes is None:
            return image_paded, masks_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale 
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale 
            return image_paded, boxes,scale, masks_paded



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True
    

def evaluate_coco(generator, model, threshold=0.05):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        oU NMSgenerator : The generator for g
        model     : The model to evaluate.
        threshold : The score threshold to use.
        
    gt_masks = gt mask in numpy uint8 
    masks = list of mask in numpy uint8
    """
    # start collecting results
    results = []
    image_ids = []
    for index in tqdm(range(len(generator))):
        img,gt_boxes,gt_labels,scale, gt_masks = generator[index]
        # run network
        scores, labels,boxes, masks  = model(img.unsqueeze(dim=0).cuda())
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        boxes /= scale
        # correct boxes for image scale
        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # compute predicted labels and scores
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                break

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.ids[index],
                'category_id' : generator.id2category[label],
                'score'       : float(score),
                'bbox'        : box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(generator.ids[index])

    if not len(results):
        return

    # write output
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    # json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

if __name__ == "__main__":
    generator=COCOGenerator(opt.test_folder,opt.anno_file,dcm=True)
    model=FCOSDetector(mode="inference")
    model = model.cuda().eval()
    model.load_state_dict(torch.load(opt.weight,map_location=torch.device('cuda')))
    evaluate_coco(generator,model)



