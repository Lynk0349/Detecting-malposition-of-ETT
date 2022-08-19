# Detecting Endotracheal Tube and Carina on Portable Supine Chest Radiographs using One-Stage Detector with a Coarse-To-Fine Attention

<style>
.red {
  color: #F60404;
}
</style>

## **Folder structure**
```
├── README.md                | 
├── environment.yaml         | environment
├── classes.txt              | 
├── coco_eval.py             | evaluate detection result in coco format
├── detect_ccy.py            | detect malposition, saving detection result and summary   
├── resnet50.pth             | 
├── run_all.py               | integrate the commands    
├── toanno.py                | generate annotations
├── train_coco.py            | training 
├── checkpoint               | save weight
------------------------------ 
├── $coco_test$              | your own dataset
│   ├── annotations          | your own annotation file
│   ├── images_dcm           | your own dicom CXRs 
│       ├── train            | 
│       ├── test             | 
│       └── val              | 
│   └── labels               | your own labels
│       ├── train            | 
│       ├── test             | 
│       └── val              |
------------------------------
├── dataset                  | 
│   ├── .DS_Store            | 
│   ├── __init__.py          | 
│   ├── augment.py           | 
│   └── COCO_dataset.py      | 
------------------------------
├── model                    | FCOS
│   ├── .DS_Store            | 
│   ├── __init__.py          | 
│   ├── config.py            | 
│   ├── fcos.py              | 
│   ├── fpn_neck.py          | neak
│   ├── head.py              | head
│   ├── loss.py              | loss function
│   ├── util.py              | some util processes and attention modules
│   └── backbone             | backbone
│       ├── __init__.py      | 
│       └── resnet.py        |
------------------------------
├── runs                     | automatically generate, tensorboard record
├── out_images               | automatically generate, detection result  
├── coco_bbox_results        | automatically generate, bbox detection result 
└── losses_ccy.xlsx          | automatically generate, summary 
```
---
## **Requirements**
```
conda env create -f environment.yaml -n $your virtual environment$
```
Change the text in the $$ to your own setting.
#### **Or install**
* pytorch == 1.9
* CUDA == 10.2
* opencv 
* pydicom
* tensorboardX
* scipy
* matplotlib
* pycocotools
* tqdm
* openpyxl
* skimage
---
#### <span class="red">Note</span>

if your CUDA version is 11.x and face the RuntimeError bellow:
```
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
```
The reason may be out of cuda memory, please down the batch size or image resolution.

## **Example**
#### **Evaluate example**
Change the text in the $$ to your own setting.
```
python coco_eval.py -t=./$coco_test$/images_dcm/test -a=./$coco_test$/annotations/instances_test_dcm.json -w=./checkpoint/final_120.pth
```
---
#### **Detection example**
Change the text in the $$ to your own setting.
```
python detect_ccy.py -t=./$coco_test$/images_dcm/test -a=./$coco_test$/annotations/instances_test_dcm.json -w=./checkpoint/final_120.pth
```
---
#### **Training example**
Change the text in the $$ to your own setting.
```
python train_coco.py -e=120 -b=2 -d=./$coco_test$/images_dcm/train -a=./$coco_test$/annotations/instances_train_dcm.json
```
---
