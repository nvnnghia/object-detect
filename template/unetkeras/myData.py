import numpy as np 
from keras.utils import Sequence
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
import cv2

# https://github.com/aleju/imgaug
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(
        iaa.OneOf([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)
        ])
    ),
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Flipud(0.5),
    iaa.Rotate((-90, 90))
],random_order=True)

class My_Generator(Sequence):

    def __init__(self, image_filenames, labels,
                 batch_size, num_cls =3, is_train=True,
                 input_size = 320, augment=True):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.input_size = input_size
        self.num_cls = num_cls

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        # return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        # batch_seg = []
        white_segs = []
        red_segs = []
        blue_segs = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread(sample)
            label_seg = img = cv2.imread(label)

            # img = cv2.resize(img, (SIZE, SIZE))
            if(self.is_augment):
                # img = seq.augment_image(img)
                img, label_seg = seq(images=[img], segmentation_maps=[label_seg])
                label_seg = label_seg[0]
                img = img[0]

            

            seg_labels = np.zeros((label_seg.shape[0], label_seg.shape[1]))
            seg_labels += ((label_seg[:, : , 0] >50)*(label_seg[:, : , 1] >50)*(label_seg[:, : , 2] >50)).astype(int) #white
            seg_labels += ((label_seg[:, : , 2] > 50 )*(label_seg[:, : , 1] <50)*(label_seg[:, : , 0] <50)).astype(int)*2 #red
            seg_labels += ((label_seg[:, : , 1] > 50 )*(label_seg[:, : , 0] <50)*(label_seg[:, : , 2] <50)).astype(int)*3 #green
            label_seg =   seg_labels

            img = cv2.resize(img, (self.input_size, self.input_size))
            label_seg = cv2.resize(label_seg, (self.input_size, self.input_size))

            label_seg[label_seg>self.num_cls]=0

            white_seg = np.zeros((label_seg.shape[0], label_seg.shape[1], 1))
            white_seg[label_seg==1] = 1
            red_seg = np.zeros((label_seg.shape[0], label_seg.shape[1], 1))
            red_seg[label_seg==2] = 1
            blue_seg = np.zeros((label_seg.shape[0], label_seg.shape[1], 1))
            blue_seg[label_seg==3] = 1

            # white_seg = np.ascontiguousarray(white_seg, dtype=np.float32)
            # red_seg = np.ascontiguousarray(red_seg, dtype=np.float32)
            # blue_seg = np.ascontiguousarray(blue_seg, dtype=np.float32)


            batch_images.append(img)
            # batch_seg.append(label_seg)
            white_segs.append(white_seg)

            red_segs.append(red_seg)
            blue_segs.append(blue_seg)

        batch_images = np.array(batch_images, np.float32) / 255
        # batch_seg = np.array(batch_seg, np.float32)
        white_segs = np.array(white_segs, np.float32)
        red_segs = np.array(red_segs, np.float32)
        blue_segs = np.array(blue_segs, np.float32)
        # if(self.is_mix):
        #     batch_images, batch_y = self.mix_up(batch_images, batch_y)
        # return batch_images, batch_seg
        return batch_images, [white_segs, red_segs, blue_segs]

    # def valid_generate(self, batch_x, batch_y):
    #     batch_images = []
    #     for (sample, label) in zip(batch_x, batch_y):
    #         img = cv2.imread(sample)
    #         img = cv2.resize(img, (SIZE, SIZE))
    #         batch_images.append(img)
    #     batch_images = np.array(batch_images, np.float32) / 255
    #     batch_y = np.array(batch_y, np.float32)
    #     return batch_images, batch_y
    
    
#test generator

if __name__ == '__main__':
    import os
    train_path = 'avmtrain.txt'
    with open(train_path, 'r') as f:
        img_files = [x.replace('/', os.sep) for x in f.read().splitlines()]

    labelseg_files = [x.replace('/images', '/labels').replace(os.path.splitext(x)[-1], '.png')
                            for x in img_files]
    mygen = My_Generator1(img_files, labelseg_files, 1, num_cls = 3, is_train=True, input_size = 320)

    for count, (x,y) in enumerate(mygen):
        print(x.shape)
        print(y[0].shape)
        image = 255*x[0]
        image[y[0][:,:,0]>0.5]=255 #mark
        image[:,:,2][y[0][:,:,1]>0.5]=255 #car
        image[:,:,0][y[0][:,:,2]>0.5]=255 #obstacle
        cv2.imwrite('draw/%s.jpg'%count, image)
        # cv2.imwrite('mask1.jpg', 255*y[0][:,:,0])
        # cv2.imwrite('mask2.jpg', 255*y[0][:,:,1])
        # cv2.imwrite('mask3.jpg', 255*y[0][:,:,2])
        if count>100:
            break
