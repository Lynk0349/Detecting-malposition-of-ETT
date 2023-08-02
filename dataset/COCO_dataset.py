import os
import cv2
import torch
import random
import pydicom
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection


def flip(img, boxes, masks):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    masks = masks.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes, masks

class COCODataset(CocoDetection):
    def __init__(self,imgs_path,anno_path,resize_size=[800,1333],is_train = True, transform=None, dcm=False):
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

        self.transform=transform
        self.resize_size=resize_size

        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]
        self.train = is_train

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
        masks = Image.fromarray(np.max(np.stack([self.coco.annToMask(o) * o["category_id"] for o in ann_mask]), axis=0))
        
        if  self.train:
            if random.random() < 0.5 :
                img, boxes, masks = flip(img, boxes, masks)
            if self.transform is not None:
                img, boxes, masks = self.transform(img, boxes, masks)
        img=np.array(img)
        masks = np.array(masks)
        img, boxes, masks = self.preprocess_img_boxes(img, boxes, masks, self.resize_size)

        classes = [o['category_id'] for o in ann_bbox]
        classes = [self.category2id[c] for c in classes]

        img=transforms.ToTensor()(img)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)
        masks = torch.LongTensor(masks).permute(2,0,1)

        return img,boxes,classes, masks

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

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def preprocess_img_boxes(self, image, boxes, masks, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        masks: [4,]
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
        masks_resized = cv2.resize(masks,(nw, nh), interpolation = cv2.INTER_NEAREST)

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized
        
        masks_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 1],dtype=np.uint8)
        masks_paded[:nh, :nw, 0] = masks_resized

        if boxes is None:
            return image_paded, masks_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes, masks_paded



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True

    def collate_fn(self,data):
        imgs_list,boxes_list,classes_list, masks_list=zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)==len(masks_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]
        
        pad_masks_list=[]
    
        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
        
            masks = masks_list[i]
            pad_masks_list.append(torch.nn.functional.pad(masks,(0,int(max_w-masks.shape[2]),0,int(max_h-masks.shape[1])),value=0.))
            

        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
        

        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)
        
        batch_masks=torch.stack(pad_masks_list)
        
    
        return batch_imgs,batch_boxes,batch_classes, batch_masks



if __name__=="__main__":
    dataset=COCODataset("./coco_ccy_f5/images_dcm/train","./coco_ccy_f5/annotations/instances_train_dcm.json", dcm=True) 
    img,boxes,classes,masks=dataset.collate_fn([dataset[0],dataset[1]])
    print(boxes,classes,masks,"\n",img.shape,boxes.shape,classes.shape, masks.shape, boxes.dtype,classes.dtype,img.dtype, masks.dtype)

