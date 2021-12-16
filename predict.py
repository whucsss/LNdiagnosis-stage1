from keras.layers import Input
from mask_rcnn import MASK_RCNN
from PIL import Image
import datetime
import os
import os.path as osp

mask_rcnn = MASK_RCNN()
count = os.listdir("./test1128")
index = 0
date = datetime.datetime.now().strftime('%Y%m%d')
for i in range(0, len(count)):
    path = os.path.join("./test1128", count[i])
    out_path = os.path.join("./predict1128", date +"_" + os.path.splitext(count[i])[0]+"_predict.jpg")
    if os.path.isfile(path) and path.endswith('jpg'):
        try:
            image = Image.open(path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            index = index+1
            image = mask_rcnn.detect_image1(image)
           # image.show()
            image.save(out_path)
            print('Saved : %s' % str(index))
print('saved all')
