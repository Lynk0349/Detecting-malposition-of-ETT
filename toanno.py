import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label_folder", type=str, default=None, help="where is your lables folder")
parser.add_argument("-c", "--classes_path", type=str, default=None, help="where is your classes.txt (.txt)")
parser.add_argument("-a", "--anno_folder", type=str, default=None, help="where is your annotations folder")
parser.add_argument("-m", "--mode", type=str, default='train', help="train/ val/ test")
opt = parser.parse_args()

def Find_center(c1,c2):
    x = round((c1[0] + c2[0])/2)
    y = round((c1[1] + c2[1])/2)
    return (x, y)

def Find_bbox(L):
    x_min = L[0][0]
    x_max = L[0][0]
    y_min = L[0][1]
    y_max = L[0][1]
    for idx in range(len(L)):
        if(L[idx][0] < x_min):
            x_min = L[idx][0]
        elif(L[idx][0] > x_max):
            x_max = L[idx][0]
        if(L[idx][1] < y_min):
            y_min = L[idx][1]
        elif(L[idx][1] > y_max):
            y_max = L[idx][1]
    return x_min, x_max, y_min, y_max

def Toanno(label_folder, classes_path, out_folder, mode='train', dcm=False, point_size=300):
    label_path = os.path.join(label_folder, mode)
    if dcm == False:
        out_path = os.path.join(out_folder, "instances_"+mode+".json")
    else:
        out_path = os.path.join(out_folder, "instances_"+mode+"_dcm.json")        
    #read_class.txt
    with open(classes_path, 'r') as cls:
        class_name = cls.read().split(" ")
    #give classes ids
    cat_info = []
    for i, cat in enumerate(class_name):
        cat_info.append({'name': cat, "id": i+1})
    #final json
    ret = {'categories':cat_info, 'images':[], 'annotations':[]}

    label_list = os.listdir(label_path)
    file_list = []
    for idx in range(len(label_list)):
        file_list.append(label_list[idx].split(".")[0])

    image_id = 0
    for image_id in range(len(file_list)):
        json_path = os.path.join(label_path, file_list[image_id]+".json")
        with open(json_path, 'r', encoding='utf-8-sig') as fj:
            data = json.load(fj)
            file_name_dcm = data['ImageFileName'].split('\\')[-1]
            temp = file_name_dcm.split(".")
            if dcm == False:
                file_name = temp[0] + ".jpg"
            else:
                file_name = temp[0] + ".dcm"
            image_info = {'file_name': file_name, 'id':int(image_id), 'width':data['Info']['Width'], 'height':data['Info']['Height']}
            # images data
            ret['images'].append(image_info)
            # key point data
            keypoint_list = []
            for point in range(len(data['KeyPoints'])):
                x, y = data['KeyPoints'][point].split(",") 
                keypoint_list.append((int(x), int(y)))
            # find annotation
            x1_min, x1_max, y1_min, y1_max = Find_bbox(keypoint_list[0:4])
            x2_min, x2_max, y2_min, y2_max = Find_bbox(keypoint_list[4:13])
            tube_point = Find_center(keypoint_list[1],keypoint_list[2])

            branch_point = keypoint_list[8]

            # ETT
            category_id = cat_info[0]['id']
            h1 = max(1, (y1_max-y1_min))
            w1 = max(1, (x1_max-x1_min))
            ann = {
                "area":h1*w1,
                "bbox":[x1_min,y1_min,w1,h1],
                "category_id":category_id,
                "id":int(len(ret['annotations'])+1),
                "image_id": image_id,
                "iscrowd":0,
                "segmentation":[[keypoint_list[0][0], keypoint_list[0][1], 
                keypoint_list[1][0], keypoint_list[1][1], 
                keypoint_list[2][0], keypoint_list[2][1], 
                keypoint_list[3][0], keypoint_list[3][1]]],
            }
            ret['annotations'].append(ann)

            # ETT tip
            category_id = cat_info[1]['id']
            ann = {
                "area":point_size*point_size,
                "bbox":[(tube_point[0]-point_size//2),(tube_point[1]-point_size//2),point_size,point_size],
                "category_id":category_id,
                "id":int(len(ret['annotations'])+1),
                "image_id": image_id,
                "iscrowd":0,
                "segmentation":[],
            }
            ret['annotations'].append(ann)

            # Branch bbox
            branch_point_list_sort_1 = sorted(keypoint_list[4:7], key = lambda s: s[1])
            branch_point_list_sort_2 = sorted(keypoint_list[7:10], key = lambda s: s[0])
            branch_point_list_sort_3 = sorted(keypoint_list[10:13], key = lambda s: s[1])
            
            pt5=branch_point_list_sort_1[0] 
            pt6=branch_point_list_sort_1[1]
            pt7=branch_point_list_sort_1[2]
            pt8=branch_point_list_sort_2[0]
            pt9=branch_point_list_sort_2[1]
            pt10=branch_point_list_sort_2[2]
            pt11=branch_point_list_sort_3[2]
            pt12=branch_point_list_sort_3[1]  
            pt13=branch_point_list_sort_3[0] 
             
            category_id = cat_info[2]['id']
            h2 = max(1, (y2_max-y2_min))
            w2 = max(1, (x2_max-x2_min))
            ann = {
                "area":h2*w2,
                "bbox":[x2_min,y2_min,w2,h2],
                "category_id":category_id,
                "id":int(len(ret['annotations'])+1),
                "image_id": image_id,
                "iscrowd":0,
                "segmentation":[[pt5[0],pt5[1],pt6[0],pt6[1],pt7[0],pt7[1],pt8[0],pt8[1],pt9[0],pt9[1],pt10[0],pt10[1],pt11[0],pt11[1],pt12[0],pt12[1],pt13[0],pt13[1]]]
            }
            ret['annotations'].append(ann)
            # Carina
            category_id = cat_info[3]['id']
            ann = {
                "area":point_size*point_size,
                "bbox":[(branch_point[0]-point_size//2),(branch_point[1]-point_size//2),point_size,point_size],
                "category_id":category_id,
                "id":int(len(ret['annotations'])+1),
                "image_id": image_id,
                "iscrowd":0,
                "segmentation":[],
            }
            ret['annotations'].append(ann)

    json.dump(ret, open(out_path, 'w'))
    
if __name__ == '__main__':   
    Toanno(opt.label_folder, opt.classes_path, opt.anno_folder, mode=opt.mode, dcm=True, point_size=300)