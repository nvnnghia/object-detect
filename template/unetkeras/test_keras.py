from myModel import *
import cv2, os
import numpy as np

IMAGE_SIZE    = 320
WEIGHTS_FINAL = 'models/model-30-0.2556.hdf5'

#define the network
model = unet()
model.load_weights(WEIGHTS_FINAL, by_name=True, skip_mismatch=True)

image_dir = 'samples1/'
imagenames = os.listdir(image_dir)
for name in imagenames:
    image =cv2.imread(image_dir+name)
    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
    np_image_data = np.asarray(image)
    np_final = np.expand_dims(np_image_data,axis=0)

    predict = model.predict(np_final)
    # print(len(predict))
    # print(predict[1].shape)


    image[:,:,1][predict[0][0].reshape(320,320)<0.5]=255 #mark
    image[:,:,2][predict[1][0].reshape(320,320)>0.5]=255 #car
    image[:,:,0][predict[2][0].reshape(320,320)>0.5]=255 #obstacle

        # cv2.imwrite('aa1.jpg',outs[2][0][1]*255)
    cv2.imwrite(os.path.join('draw',name.split('/')[-1]), image)

    # cv2.imwrite('w.jpg',predict[1][0]*255)
    # break


