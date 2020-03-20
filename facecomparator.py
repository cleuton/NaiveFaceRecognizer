import cv2
import sys
import numpy as np
import math

#pip install openface -  https://github.com/cmusatyalab/openface/blob/master/demos/compare.py

embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")

def extractEmbeddings(image):
    faceBlob = cv2.dnn.blobFromImage(image, 1.0, \
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec=embedder.forward()
    return vec

def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))

if __name__=="__main__":
    imfile1 = sys.argv[1]
    print(imfile1)
    img1=cv2.imread(imfile1)
    emb1=extractEmbeddings(img1)
    print("Type:",type(emb1))
    print("Shape:",emb1.shape)
    #print(emb1[0])
    imfile2 = sys.argv[2]
    print(imfile2)
    img2=cv2.imread(imfile2)
    emb2=extractEmbeddings(img2)
    print("Type:",type(emb2))
    print("Shape:",emb2.shape)
    #print(emb2[0])    
    distance = euclidean_dist(emb1[0],emb2[0])
    print("Distance:",distance)

    



