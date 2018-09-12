"""Unit Test for VGG19 initialize and build image sequence"""

import tensorflow as tf

import utils
from vgg.vgg import VGG19

def test1(init_img, styl_layers, cont_layers):
    # initial a model for each activity

    with tf.Session() as sess:
        gen_styl_act = VGG19().build(init_img).layer_dict(styl_layers)
        gen_cont_act = VGG19().build(init_img).layer_dict(cont_layers)
     
        test1_styl = {}
        for l in styl_layers:
            test1_styl[l] = sess.run(gen_styl_act[l])
            #print(l, "\n", sess.run(gen_styl_act[l]))
            
        test1_cont = {}
        for l in cont_layers:
            test1_cont[l] = sess.run(gen_cont_act[l])
            #print(l, '\n', sess.run(gen_cont_act[l]))
    return test1_styl, test1_cont


def test2(init_img, styl_layers, cont_layers):
    # initial one model, then build image for each activity 

    with tf.Session() as sess:
        vgg_model = VGG19()
        gen_styl_act = vgg_model.build(init_img).layer_dict(styl_layers)
        gen_cont_act = vgg_model.build(init_img).layer_dict(cont_layers)
        
        test2_styl = {}
        for l in styl_layers:
            test2_styl[l] = sess.run(gen_styl_act[l])
        
        test2_cont = {}
        for l in cont_layers:
            test2_cont[l] = sess.run(gen_cont_act[l])
    return test2_styl, test2_cont


def test3(init_img, styl_layers, cont_layers):
    # initial one model, one build image for each activity 

    with tf.Session() as sess:
        vgg_model = VGG19().build(init_img)
        gen_styl_act = vgg_model.layer_dict(styl_layers)
        gen_cont_act = vgg_model.layer_dict(cont_layers)
        
        test3_styl = {}
        for l in styl_layers:
            test3_styl[l] = sess.run(gen_styl_act[l])
        
        test3_cont = {}
        for l in cont_layers:
            test3_cont[l] = sess.run(gen_cont_act[l])
    return test3_styl, test3_cont
