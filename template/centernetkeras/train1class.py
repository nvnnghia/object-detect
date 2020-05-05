import cv2, os
from PIL import Image, ImageDraw
import numpy as np 
from keras.utils import Sequence
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from PIL import Image
from keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from keras.models import Model
from keras.objectives import mean_squared_error
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from keras.optimizers import Adam, RMSprop, SGD

###
category_n=1
output_layer_n=category_n+4


####FOR AUGMENTATION#####
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(
        iaa.OneOf([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)
        ])
    ),
    # iaa.AdditiveGaussianNoise(scale=(0, 0.08*255)),
    # sometimes(iaa.Rotate((-90, 90))),
    sometimes(iaa.Fliplr(0.5))
    # sometimes(iaa.Crop(percent=(0, 0.2)))
    # sometimes(iaa.Flipud(0.5))
    
],random_order=False)

def augment(img):
    return seq(images=[img])

#####FOR DATASET######
class My_generator(Sequence):
    def __init__(self, img_paths, label_paths, batch_size, input_size=512, is_train = False):
        self.imagenames = img_paths
        self.labels = label_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = self.input_size//4
        self.is_train = is_train
        if self.is_train:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.imagenames)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.imagenames[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

        if self.is_train:
            return self.train_generator(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.imagenames, self.labels = shuffle(self.imagenames, self.labels)


    def train_generator(self, batch_x, batch_y):
        batch_imgs = []
        batch_segs = []
        output_height,output_width=self.output_size,self.output_size
        for (img_path, label_path) in zip(batch_x, batch_y):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_size, self.input_size))

            # img_h, img_w = img.shape[:2]

            # output_height,output_width=104,104

            #PROCESS LABELS
            output_layer=np.zeros((output_height,output_width,(output_layer_n+category_n)))
            file1 = open(label_path)
            lines = file1.readlines()
            file1.close()

            for line in lines:
                cls, xc, yc, w, h = map(float,line.strip().split(' '))

                x_c, y_c, width, height = xc*output_height, yc*output_height, w*output_height, h*output_height
                # print(x_c, y_c)

                category=int(cls) #not classify, just detect
                heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)
                                    *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))
                output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:])

                output_layer[int(y_c//1),int(x_c//1),category_n+category]=1
                output_layer[int(y_c//1),int(x_c//1),2*category_n]=y_c%1#height offset
                output_layer[int(y_c//1),int(x_c//1),2*category_n+1]=x_c%1
                output_layer[int(y_c//1),int(x_c//1),2*category_n+2]=height/output_height
                output_layer[int(y_c//1),int(x_c//1),2*category_n+3]=width/output_width


            #     xmin = int((x_c-width/2))
            #     ymin = int((y_c-height/2))
            #     xmax = int((x_c+width/2))
            #     ymax = int((y_c+height/2))
            #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
            #     cv2.putText(img, str(cls), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),1,cv2.LINE_AA)

            # cv2.imwrite('draw/'+img_path.split('/')[-1],output_layer[:,:,0]*255)
            # cv2.imwrite('draw/a'+img_path.split('/')[-1],img)


            # print(sample, img.shape)
            # img = augment(img)
            # print(output_layer.shape)
            # print(img.shape)
            images_aug, segmaps_aug = seq(images=[img], heatmaps=[output_layer.astype(np.float32)])


            batch_imgs.append(images_aug[0])
            batch_segs.append(segmaps_aug[0])
            # batch_imgs.append(img[0])
            # batch_segs.append(output_layer)


        batch_imgs = np.array(batch_imgs, np.float32) /255
        batch_segs = np.array(batch_segs, np.float32)

        return batch_imgs, batch_segs


    def valid_generate(self, batch_x, batch_y):
        batch_imgs = []
        batch_segs = []
        output_height,output_width=self.output_size,self.output_size
        for (img_path, label_path) in zip(batch_x, batch_y):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_size, self.input_size))

            #PROCESS LABELS
            output_layer=np.zeros((output_height,output_width,(output_layer_n+category_n)))
            file1 = open(label_path)
            lines = file1.readlines()
            file1.close()

            for line in lines:
                cls, xc, yc, w, h = map(float,line.strip().split(' '))

                x_c, y_c, width, height = xc*output_height, yc*output_height, w*output_height, h*output_height

                category=int(cls) #not classify, just detect
                heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)
                                    *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))
                output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:])

                output_layer[int(y_c//1),int(x_c//1),category_n+category]=1
                output_layer[int(y_c//1),int(x_c//1),2*category_n]=y_c%1#height offset
                output_layer[int(y_c//1),int(x_c//1),2*category_n+1]=x_c%1
                output_layer[int(y_c//1),int(x_c//1),2*category_n+2]=height/output_height
                output_layer[int(y_c//1),int(x_c//1),2*category_n+3]=width/output_width

            batch_imgs.append(img)
            batch_segs.append(output_layer)


        batch_imgs = np.array(batch_imgs, np.float32) /255
        batch_segs = np.array(batch_segs, np.float32)

        return batch_imgs, batch_segs

def test_dataset():
    train_path = 'bdd_coco_human.txt'
    with open(train_path, 'r') as f:
        img_files = [x.replace('/', os.sep) for x in f.read().splitlines()]

    label_files = [x.replace(os.path.splitext(x)[-1], '.txt') for x in img_files]

    mygen = My_generator(img_files, label_files, batch_size=1, is_train=True, input_size = 416)

    for count, (x,y) in enumerate(mygen):
        # print(x.shape)
        # print(y.shape)
        x = x[0]*255
        y= y[0]

        points = np.argwhere(y[:,:,1] ==1)
        for y1,x1 in points:
            # print(x1,y1)
            offsety = y[:,:,2][y1,x1]
            offetx = y[:,:,3][y1,x1]
            
            h = y[:,:,4][y1,x1]*104
            w = y[:,:,5][y1,x1]*104

            x1, y1 = x1 + offetx, y1+offsety 

            xmin = int((x1-w/2)*4)
            xmax = int((x1+w/2)*4)
            ymin = int((y1-h/2)*4)
            ymax = int((y1+h/2)*4)

            cv2.rectangle(x, (xmin, ymin), (xmax, ymax), (0,255,255), 2)


            cv2.circle(x, (int(x1*4),int(y1*4)), 5, (0,0,255), 2) 

        cv2.imwrite('draw/'+str(count)+'.jpg',y[:,:,1]*255)
        cv2.imwrite('draw/a'+str(count)+'.jpg',x)

        if count>10:
            break

# test_dataset()

######LOSSES FUNCTION#####
def all_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    alpha=2.
    beta=4.

    heatmap_true_rate = K.flatten(y_true[...,:category_n])
    heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])
    heatmap_pred = K.flatten(y_pred[...,:category_n])
    heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))
    offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))
    sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))
    
    all_loss=(heatloss+1.0*offsetloss+5.0*sizeloss)/N
    return all_loss

def size_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))
    return (5*sizeloss)/N

def offset_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))
    return (offsetloss)/N
  
def heatmap_loss(y_true, y_pred):
    mask=K.sign(y_true[...,2*category_n+2])
    N=K.sum(mask)
    alpha=2.
    beta=4.

    heatmap_true_rate = K.flatten(y_true[...,:category_n])
    heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])
    heatmap_pred = K.flatten(y_pred[...,:category_n])
    heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))
    return heatloss/N


##########MODEL#############

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
  x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
  x_deep = BatchNormalization()(x_deep)   
  x_deep = LeakyReLU(alpha=0.1)(x_deep)
  x = Concatenate()([x_shallow, x_deep])
  x=Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
  x = BatchNormalization()(x)   
  x = LeakyReLU(alpha=0.1)(x)
  return x
  


def cbr(x, out_layer, kernel, stride):
  x=Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.1)(x)
  return x

def resblock(x_in,layer_n):
  x=cbr(x_in,layer_n,3,1)
  x=cbr(x,layer_n,3,1)
  x=Add()([x,x_in])
  return x  


#I use the same network at CenterNet
def create_model(input_shape, aggregation=True):
    input_layer = Input(input_shape)
    
    #resized input
    input_layer_1=AveragePooling2D(2)(input_layer)
    input_layer_2=AveragePooling2D(2)(input_layer_1)

    #### ENCODER ####

    x_0= cbr(input_layer, 16, 3, 2)#512->256
    concat_1 = Concatenate()([x_0, input_layer_1])

    x_1= cbr(concat_1, 32, 3, 2)#256->128
    concat_2 = Concatenate()([x_1, input_layer_2])

    x_2= cbr(concat_2, 64, 3, 2)#128->64
    
    x=cbr(x_2,64,3,1)
    x=resblock(x,64)
    x=resblock(x,64)
    
    x_3= cbr(x, 128, 3, 2)#64->32
    x= cbr(x_3, 128, 3, 1)
    x=resblock(x,128)
    x=resblock(x,128)
    x=resblock(x,128)
    
    x_4= cbr(x, 256, 3, 2)#32->16
    x= cbr(x_4, 256, 3, 1)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
 
    x_5= cbr(x, 512, 3, 2)#16->8
    x= cbr(x_5, 512, 3, 1)
    
    x=resblock(x,512)
    x=resblock(x,512)
    x=resblock(x,512)
    
    #### DECODER ####
    x_1= cbr(x_1, output_layer_n, 1, 1)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_2= cbr(x_2, output_layer_n, 1, 1)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3= cbr(x_3, output_layer_n, 1, 1)
    x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n) 
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

    x_4= cbr(x_4, output_layer_n, 1, 1)

    x=cbr(x, output_layer_n, 1, 1)
    x= UpSampling2D(size=(2, 2))(x)#8->16 

    x = Concatenate()([x, x_4])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#16->32

    x = Concatenate()([x, x_3])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#32->64 

    x = Concatenate()([x, x_2])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)#64->128 

    x = Concatenate()([x, x_1])
    x=Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
    out = Activation("sigmoid")(x)
    
    model=Model(input_layer, out)
    
    return model
  

#####TRAIN##########
def lrs(epoch):
    lr = 0.001
    if epoch >= 20: lr = 0.0002
    return lr
    
def train(train_path='bdd_coco_human.txt', batch_size=32, input_size=320, n_epoch=30):
    learning_rate=0.001

    with open(train_path, 'r') as f:
        img_files = [x.replace('/', os.sep) for x in f.read().splitlines()]

    label_files = [x.replace(os.path.splitext(x)[-1], '.txt') for x in img_files]

    train_list, valid_list, train_label, valid_label = train_test_split(img_files, label_files, test_size=0.15, random_state=8)


    mygen = My_generator(train_list, train_label, batch_size=batch_size, is_train=True, input_size = input_size)
    myval = My_generator(valid_list, valid_label, batch_size=batch_size, is_train=False, input_size = input_size)


    model=create_model(input_shape=(input_size,input_size,3))
    
    lr_schedule = LearningRateScheduler(lrs)


    # EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 60, verbose = 1)
    # ModelCheckpoint
    weights_dir = 'models/'

    if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
    model_checkpoint = ModelCheckpoint(weights_dir + "{epoch:02d}-{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,
                                          save_best_only = False, save_weights_only = True, period = 3)
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 4, verbose = 1)
    

    print(model.summary())

    model.compile(loss=all_loss, optimizer=Adam(lr=learning_rate), metrics=[heatmap_loss,size_loss,offset_loss])

    hist = model.fit_generator(
        mygen,
        steps_per_epoch = len(label_files) // batch_size,
        epochs = n_epoch,
        validation_data=myval,
        validation_steps = len(valid_list) // batch_size,
        callbacks = [early_stopping, reduce_lr, model_checkpoint],
        shuffle = True,
        verbose = 1
    )
    model.save_weights('final_weights.h5')

    print(hist.history.keys())
    np.save('his.npy', hist.history)




#######PREDICT IMAGES###########
from postprocess import *
from matplotlib import pyplot as plt
def predict_image(imagepath, input_size=320, weights_file=''):
    # input_size = 320
    # weights_file = ''

    pred_out_h=int(input_size/4)
    pred_out_w=int(input_size/4)

    model=create_model(input_shape=(input_size,input_size,3))

    model.load_weights(weights_file,by_name=True, skip_mismatch=True)

    # for i in np.arange(0,1):
    # img = np.asarray(Image.open(imagepath).resize((input_size,input_size)).convert('RGB'))
    # predict=model.predict((img.reshape(1,input_size,input_size,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))
    img=Image.open(imagepath).convert("RGB")
    predict=model.predict((np.asarray(img.resize((input_size,input_size))).reshape(1,input_size,input_size,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))


    # img = cv2.imread(imagepath)
    # img = cv2.resize(img, (input_size, input_size))
    # predict = model.predict((img[np.newaxis])/255).reshape(pred_out_h,pred_out_w,(category_n+4))

    box_and_score=NMS_all(predict,category_n, pred_out_h, pred_out_w, score_thresh=0.3,iou_thresh=0.4)
    if len(box_and_score)==0:
        print('no boxes found!!')
        return

    heatmap=predict[:,:,0]
    print(img.size)
    print_w, print_h = img.size
    box_and_score=box_and_score*[1,1,print_h/pred_out_h,print_w/pred_out_w,print_h/pred_out_h,print_w/pred_out_w]
    img=draw_rectangle(box_and_score[:,2:],img,"red")
    # img=draw_rectangle(true_boxes,img,"blue")

    fig, axes = plt.subplots(1, 2,figsize=(15,15))
    #axes[0].set_axis_off()
    axes[0].imshow(img)
    #axes[1].set_axis_off()
    axes[1].imshow(heatmap)#, cmap='gray')
    #axes[2].set_axis_off()
    #axes[2].imshow(heatmap_1)#, cmap='gray')
    plt.show()


if __name__=='__main__':
    # train(train_path='bdd_coco_human.txt', batch_size=32, input_size=320, n_epoch=30)

    predict_image('testimages/260.jpg', input_size=512, weights_file='models/21-2.270.hdf5')