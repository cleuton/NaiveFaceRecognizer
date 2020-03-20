import facecomparator
import sys, os
import json
import cv2
import numpy

path_processed="./test"

# Read json faces db
with open('./faces.json') as f:
  persons_dict = json.load(f)

def recognize(image):
    embeddings=facecomparator.extractEmbeddings(image)
    minor_distance = 999.99
    name = ""
    for person in persons_dict["persons"]:       
        person_embeddings = numpy.asarray(json.loads(person["embeddings"]))
        distance = facecomparator.euclidean_dist(embeddings[0],person_embeddings[0])
        print("Comparing with:",person["name"],"distance:",distance)
        if distance <= minor_distance:
            minor_distance = distance
            name = person["name"]
            print("Chosen:",name)
    return minor_distance,name

for filename in os.listdir(path_processed):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        fullpath = os.path.join(path_processed,filename)
        print("Processing:", fullpath)
        image = cv2.imread(fullpath)
        distance,name = recognize(image)
        cv2.imshow(name + " (" + str(distance) + ")",image)
cv2.waitKey(0)
