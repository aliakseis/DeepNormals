import os, argparse

#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))

#tf.keras.backend.get_session().run(tf.global_variables_initializer())

import cv2

from model import *

keras = tf.keras

from keras import backend as K

K.set_learning_phase(0)

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 

    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    #absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    #output_graph = absolute_model_dir + "/frozen_model.pb"
    output_graph = "./frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)


        # We restore the weights
        #saver.restore(sess, input_checkpoint)
        saver = GenerateNet()
        saver.load('Net/DeepNormals/DeepNormals')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(output_graph_def)

        #optimized_graph_def = fold_batch_norms(optimized_graph_def)
        #optimized_graph_def = fuse_resize_and_conv(optimized_graph_def, output_node_names)

        # Finally we serialize and dump the output graph to the filesystem
        #with tf.gfile.GFile(output_graph, "wb") as f:
        #    f.write(output_graph_def.SerializeToString())
        f = open(output_graph, "wb")
        f.write(output_graph_def.SerializeToString())
        f.close()
        print("%d ops in the final graph." % len(output_graph_def.node))
        cv2.dnn.writeTextGraph(output_graph, './graph.pbtxt')

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)
