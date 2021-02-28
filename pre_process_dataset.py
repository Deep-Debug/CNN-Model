import cv2,os
import numpy as numpy
import csv
import glob
import random
label = "Normal"
dirList = glob.glob("The IQ-OTHNCCD lung cancer dataset/"+label+"/*.jpg")
random.shuffle(dirList)
print(len(dirList))
count=0
train=0
valid=0

for img_path in dirList:
    if count<375:
        dest_path="The IQ-OTHNCCD lung cancer dataset/train/Normal/"    
        im=cv2.imread(img_path)
        im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        li=img_path.split("\\")
        dest_path=dest_path+li[1]
        
        op=cv2.resize(im_gray,dsize=(50,50))
        cv2.imwrite(dest_path,op)
        train+=1
        count+=1
        print(dest_path)
    else:
        dest_path="The IQ-OTHNCCD lung cancer dataset/valid/Normal/"    
        im=cv2.imread(img_path)
        im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        li=img_path.split("\\")
        dest_path=dest_path+li[1]
        
        op=cv2.resize(im_gray,dsize=(50,50))
        cv2.imwrite(dest_path,op)
        valid+=1
        count+=1
        print(dest_path)
print("finished")
print("total train images=",train)
print("total valid images=",valid)
print("total images=",count)
        


