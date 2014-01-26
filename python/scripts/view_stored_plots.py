
import numpy as np
import cv2

from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('id')
args = parser.parse_args()

client = MongoClient()
db = client.timb_experiments_db
fs = gridfs.GridFS(db)

o = db.experiments.find_one({'_id': ObjectId(args.id)})

from pprint import pprint
pprint(o['tracker_params'])

i = 0
length = len(o['log'])
while True:
  entry = o['log'][i]
  img = cv2.imdecode(np.frombuffer(fs.get(entry['plot_file']).read(), dtype=np.uint8), 1)
  cv2.imshow('image', img)

  key = cv2.waitKey(0) & 0xff
  if key == 27: # esc
    break
  elif key == 81: # left arrow
    i = (i - 1) % length
  elif key == 83: # right arrow
    i = (i + 1) % length
