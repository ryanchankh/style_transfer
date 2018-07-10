from datetime import datetime
import tensorflow as tf

from style_transfer2 import StyleTransfer
from utils import load_image, save_image

class OPTIONS():


    init_time = datetime.now().strftime("%H%M%S_%Y%m%d")

    # image paths
    styl_img_path = './images/style/van_gough.jpg'
    cont_img_path = './images/content/dom.jpg'
    white_img_path = './images/others/plain_white.jpg'
    gen_img_path = './gen_img/' + init_time + '.jpeg'

    # hyper-parameters
    #img_shape = utils.optimal_dimension(cont_img_file, styl_img_file)
    img_shape = (1, 224, 224, 3) # [batch, height, width, channels]
    alpha = 1           # style weight alpha
    beta = 0.1         # content weight beta
    total_var = 1.0     # total variation weight
    l_rate = 10         # learning rate
    num_steps = 100     # training iterations

   # content and style layers used in style transfer
    cont_layers = ["conv2_2"]
    styl_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # weights on each style layer
    styl_weights = {"conv1_1": 0.2, "conv1_2": 0.2, "pool1": 0,
                    "conv2_1": 0.2, "conv2_2": 0, "pool2": 0,
                    "conv3_1": 0.2, "conv3_2": 0, "conv3_3": 0.2, "conv3_4": 0, "pool3": 0,
                    "conv4_1": 0.2, "conv4_2": 0, "conv4_3": 0.2, "conv4_4": 0, "pool4": 0,
                    "conv5_1": 0.2, "conv5_2": 0, "conv5_3": 0.2, "conv5_4": 0, "pool5": 0}


# turn image into numpy arrays
styl_img = load_image(OPTIONS.styl_img_path, shape=OPTIONS.img_shape)
cont_img = load_image(OPTIONS.cont_img_path, shape=OPTIONS.img_shape)
init_img = load_image(OPTIONS.white_img_path, shape=OPTIONS.img_shape)


model = StyleTransfer(init_img,
                      OPTIONS.img_shape,
                      OPTIONS.cont_layers,
                      OPTIONS.styl_layers,
                      OPTIONS.styl_weights,
                      OPTIONS.alpha,
                      OPTIONS.beta,
                      OPTIONS.l_rate)


with tf.Session(graph=model.graph) as sess:

    sess.run(tf.global_variables_initializer())
    feed_dict = {model.cont_img: cont_img, model.styl_img: styl_img}

    for step in range(OPTIONS.num_steps):
        _, gen_img, loss = sess.run([model.train, model.image, model.total_loss], feed_dict=feed_dict)

        if step % 10 == 0:
            print("Step: {}\tloss: {}".format(step, loss))

            save_image(OPTIONS.gen_img_path, gen_img, OPTIONS.img_shape)
    save_image(OPTIONS.gen_img_path, gen_img, OPTIONS.img_shape)
