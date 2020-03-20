import pythonfaces
import facecomparator
import sys
import os
import json
from json import JSONEncoder
import cv2
import numpy

path_originals="./original"
path_processed="./processed"
path_json="./"


persons = []

def process_file(fname):
    faceslist = pythonfaces.imageFromFile(fname,img_h=512,img_w=512) 
    fcount=0
    for bwimage in faceslist:
        image = backtorgb = cv2.cvtColor(bwimage,cv2.COLOR_GRAY2RGB)
        fcount += 1
        pname = os.path.splitext(os.path.basename(fname))[0]
        extension = os.path.splitext(os.path.basename(fname))[1]
        oname = pname + "." + str(fcount) + extension
        oname = os.path.join(path_processed,oname)
        print("saving:",oname)
        cv2.imwrite(oname,image)
        face_embeddings = facecomparator.extractEmbeddings(image)
        encoded_embeddings = json.dumps(face_embeddings, cls=NumpyArrayEncoder)     
        person = {"name" : pname, "embeddings" : encoded_embeddings }
        persons.append(person)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

for filename in os.listdir(path_originals):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        fullpath = os.path.join(path_originals,filename)
        print(fullpath)
        process_file(fullpath)

persons_dict = {"persons": persons}
jsondb = os.path.join(path_json,"faces.json")

# Write the json faces database with 128D encodings and names for each face
with open(jsondb, 'w') as json_file:
  json.dump(persons_dict, json_file)

