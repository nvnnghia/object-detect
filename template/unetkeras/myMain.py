from myModel import *
from myData import *
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE =16
NET_INPUT_SIZE = 320
EPOCHS = 30
if __name__ == '__main__':
    train_path = 'avmtrain.txt'
    with open(train_path, 'r') as f:
        img_files = [x.replace('/', os.sep) for x in f.read().splitlines()]

    labelseg_files = [x.replace('/images', '/labels').replace(os.path.splitext(x)[-1], '.png')
                            for x in img_files]

    train_generator = My_Generator(img_files, labelseg_files, BATCH_SIZE, num_cls = 3, is_train=True, input_size = NET_INPUT_SIZE)

    # create callbacks list


    weight_path = "models/model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, 
                                 save_best_only=False, mode='min', save_weights_only = True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, 
                                       verbose=1, mode='min', epsilon=0.0001)
    early = EarlyStopping(monitor="loss", 
                      mode="min", 
                      patience=9)

    callbacks_list = [checkpoint, reduceLROnPlat, early]

    model = unet()

    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(img_files)) / float(BATCH_SIZE)),
        # validation_data=valid_generator,
        # validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),
        epochs=EPOCHS,
        verbose=1,
        workers=1, use_multiprocessing=False,
        callbacks=callbacks_list)

# # convert to tenorflow and save model as .pb file
# pred_node_names = [None]
# pred = [None]
# for i in range(1):
#     pred_node_names[i] = "output_node"+str(i)
#     pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])

# sess = K.get_session()
# constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
# graph_io.write_graph(constant_graph, ".", "model1.pb", as_text=False)