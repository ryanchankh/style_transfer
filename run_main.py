"""
Style Tranfser 
"""

import tensorflow as tf
import numpy as np 
from PIL import Image
import time
from datetime import datetime

import vgg19.vgg as vgg
import utils
import style_transfer as st


print(datetime.now(), "Loading all hyperparameters. ")



# hyper-parameters
img_shape = np.array([1, 100, 100, 3]) # [batch, height, width, channels]
styl_loss_alpha, cont_loss_beta = 50, 0.05 # alpha/beta = 1e-3 or 1e-4
learning_rate = 10
num_steps = 100 # training iterations

# image paths
styl_img_file = 'style/starry_night.jpg'
cont_img_file = 'content/tubingen.jpg'
white_img_file = 'white_noise/plain_white.jpg'
gen_img_file = 'gen_img/gen_img.jpeg'

# turn image into numpy arrays
print(datetime.now(), "Loading images")
styl_img = utils.load_image(styl_img_file, shape=img_shape) 
cont_img = utils.load_image(cont_img_file, shape=img_shape) 
gen_img = utils.load_image(white_img_file, shape=img_shape)
print("\n")

# content and style layers used in style transfer
cont_layers = ["conv4_2"]
styl_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
gen_layers = cont_layers + styl_layers

# weights on each style layer
styl_layer_weights = {"conv1_1": 0.2, "conv1_2": 0, "pool1": 0, 
                      "conv2_1": 0.2, "conv2_2": 0, "pool2": 0,
                      "conv3_1": 0.2, "conv3_2": 0, "conv3_3": 0, "conv3_4":0, "pool3": 0, 
                      "conv4_1": 0.2, "conv4_2": 0, "conv4_3":0, "conv4_4": 0, "pool4": 0,
                      "conv5_1":0.2, "conv5_2": 0, "conv5_3": 0, "conv5_4": 0, "pool5": 0}

# initialize Sytle_Transfer Model
print(datetime.now(), "Initializing Style Transfer Model")
time_model_start = time.time()
model = st.StyleTransfer(gen_img,
                         img_shape, 
                         cont_layers, 
                         styl_layers, 
                         styl_layer_weights,
                         styl_loss_alpha,
                         cont_loss_beta,
                         learning_rate)
time_model_end = time.time()
print("Time taken: ", time_model_end - time_model_start)

# begin style transfer
with tf.Session(graph=model.graph) as sess:
    
    # initialize variables
    sess.run(model.init_op, feed_dict={model.cont_img:cont_img, model.styl_img:styl_img})
    
    # style and content activities 
    styl_activity_calc = sess.run(model.styl_activity_calc, feed_dict={model.styl_img: styl_img})
    cont_activity_calc = sess.run(model.cont_activity_calc, feed_dict={model.cont_img: cont_img})
    
    # training
    print("Begin Training.")
    time_train_start = time.time()
    for step in range(num_steps):

        # compute gradient and add to gen_img
        train_dict = {}
        train_dict.update({model.styl_activity[ln]:styl_activity_calc[ln] for ln in styl_layers})
        train_dict.update({model.cont_activity[ln]:cont_activity_calc[ln] for ln in cont_layers})
        _, total_loss, styl_loss, cont_loss, gen_img = sess.run([model.train_ops, model.total_loss, model.styl_loss, model.cont_loss, model.image], feed_dict=train_dict)

        # print_step
        if step % 10 == 0:
            print("Step: {}/{}".format(step, num_steps))
            print("styl_loss:", styl_loss)
            print("cont_loss:", cont_loss)
            print("total_loss:", total_loss)
            print()

            # save_image("./step_log/" + str(step) + ".jpg", gen_img)
        
    time_train_end = time.time()
    print("Training Completed.")
    print("Training time required: ", time_model_end - time_model_start)
    
    # save result to gen_img_file
    utils.save_image(gen_img_file, gen_img)

    # tensorboard graph
    #tf.summary.FileWriter("./log", model.graph)


  
