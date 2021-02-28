import cv2,os
import numpy as numpy
import csv
import glob

label="Normal"

dirList=glob.glob("The IQ-OTHNCCD lung cancer dataset/valid/"+label+"/*.jpg")
#print(dirList)
file=open("cancer_dataset/test.csv","a")
for img_path in dirList:
    
    im=cv2.imread(img_path)
    
    im=cv2.GaussianBlur(im,(5,5),2)
    
    im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    ret,thresh=cv2.threshold(im_gray,127,255,0)
    
    contours,_=cv2.findContours(thresh,1,2)
    
    file.write(label)
    file.write(",")
    im_gray=im_gray.flatten()
    
    for i in im_gray:
        file.write(str(i))
        file.write(',')
    file.write("\n")


    
print("finished")