import os

# make annotations
os.system("python toanno.py -l=./coco_test/labels -c=./classes.txt -a=./coco_test/annotations -m=train")

# train
os.system("python train_coco.py -e=120 -b=2 -d=./coco_test/images_dcm/train -a=./coco_test/annotations/instances_train_dcm.json")

# evaluate
os.system("python coco_eval.py -t=./coco_test/images_dcm/test -a=./coco_test/annotations/instances_test_dcm.json -w=./checkpoint/final_1.pth")

# detect
os.system("python detect_ccy.py -t=./coco_test/images_dcm/test -a=./coco_test/annotations/instances_test_dcm.json -w=./checkpoint/final_120.pth")