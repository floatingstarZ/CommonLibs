import os
import json
import cv2
from
fpath = './annotation_val.odgt'
def load_file(fpath):#fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def crowdhuman2coco(odgt_path, img_folder, json_path):  # 一个输入文件路径，一个输出文件路径
    records = load_file(odgt_path)  # 提取odgt文件数据
    # 预处理




crowdhuman2coco('D:/DataBackup/CrowdHuman/annotation_val.odgt',
                'D:/DataBackup/CrowdHuman/Images',
                'D:/DataBackup/CrowdHuman/annotation_coco_val.json')

crowdhuman2coco('D:/DataBackup/CrowdHuman/annotation_train.odgt',
                'D:/DataBackup/CrowdHuman/Images',
                'D:/DataBackup/CrowdHuman/annotation_coco_train.json')