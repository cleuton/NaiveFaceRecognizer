import pythonfaces
import cv2
import os
import sys

path_test="./test"
path_source="./test_source"

def process_file(fname):
    faceslist = pythonfaces.imageFromFile(fname,img_h=512,img_w=512) 
    fcount=0
    for bwimage in faceslist:
        image = backtorgb = cv2.cvtColor(bwimage,cv2.COLOR_GRAY2RGB)
        fcount += 1
        pname = os.path.splitext(os.path.basename(fname))[0]
        extension = os.path.splitext(os.path.basename(fname))[1]
        oname = pname + "." + str(fcount) + extension
        oname = os.path.join(path_test,oname)
        print("saving:",oname)
        cv2.imwrite(oname,image) 

for filename in os.listdir(path_source):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        fullpath = os.path.join(path_source,filename)
        print("Processing:",fullpath)
        process_file(fullpath)