import tensorflow as tf # Default graph is initialized when the library is imported
import os
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import cv2

# if 1:
with tf.Graph().as_default() as graph: # Set default graph as graph
    sess = tf.Session()
    # Load the graph in graph_def
    print("load graph")

    # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
    with gfile.FastGFile("parking.pb",'rb') as f:

        # Set FCN graph to the default graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

        # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

        tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name="",
        op_dict=None,
        producer_op_list=None
        )

        # Print the name of operations in the session
        for op in graph.get_operations():
                print "Operation Name :",op.name         # Operation name
                print "Tensor Stats :",str(op.values())     # Tensor name

        # INFERENCE Here
        l_input = graph.get_tensor_by_name('input_1:0') # Input Tensor
        l_output = graph.get_tensor_by_name('output_node0:0') # Output Tensor

        print "Shape of input : ", tf.shape(l_input)
        #initialize_all_variables
        tf.global_variables_initializer()

                                # # Run Kitty model on single image

for i in range(10):
    start = time.time()
    print("Load Image...")
    # Read the image & get statstics
    # image = scipy.misc.imread('test.jpg')
    image = cv2.imread('test.jpg')
    image = image.astype(float)
    image = image/255

    Session_out = sess.run( l_output, feed_dict = {l_input : [image]} )
    # print(type(Session_out))
    # print(Session_out.shape)
    print(time.time() - start)
np.save('out.npy', Session_out)
