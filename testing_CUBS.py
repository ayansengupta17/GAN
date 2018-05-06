import argparse
from os import listdir
from os.path import isfile, join
import skipthoughts
import os.path
import pickle
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
model = skipthoughts.load_model()


beta=0.3 #Leak Factor
keep_prob=0.7 #Dropout layer parameter
noise_length=100 #Noise vector length
batch_size=1 # Batch Size for training
lam=2.0 #divergence loss lagrange multiplier
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)


def encoding(input):

    encoded_vector_dir = './encoded_vector'
    try:
        caption_vectors = skipthoughts.encode(model, input)
        print("Sentence encoding sucessful")
    except:
        print("Failed sentence encoding")
    files_ = 'test_vector.pkl'
    with open(join(encoded_vector_dir,files_), mode='wb') as myfile:
        pickle.dump(caption_vectors, myfile)

def Generator(batch_size,z_len,encoded_sentences_tensor,reuse_flag,training_flag):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_flag):

        g_mu_w_0=tf.get_variable('g_mu_w_0',shape=[4800,128],initializer=tf.contrib.layers.xavier_initializer())
        g_mu_b_0=tf.get_variable('g_mu_b_0',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_mu_a_0=tf.matmul(encoded_sentences_tensor,g_mu_w_0)+g_mu_b_0
        g_mu_a_0=tf.maximum(g_mu_a_0,beta*g_mu_a_0)

        g_sd_w_0=tf.get_variable('g_sd_w_0',shape=[4800,128],initializer=tf.contrib.layers.xavier_initializer())
        g_sd_b_0=tf.get_variable('g_sd_b_0',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_sd_a_0=tf.matmul(encoded_sentences_tensor,g_sd_w_0)+g_sd_b_0
        g_sd_a_0=tf.maximum(g_sd_a_0,beta*g_sd_a_0)

        epsilon=tf.random_normal([batch_size,128],mean=0,stddev=1)
        g_cond=g_mu_a_0+tf.multiply(g_sd_a_0,epsilon)

        z=tf.random_normal([batch_size,z_len],mean=0,stddev=1)
        ip=tf.concat([z,g_cond],1)

        g_w1=tf.get_variable('g_w1',shape=[z_len+128,4*4*1024],initializer=tf.contrib.layers.xavier_initializer())
        g_b1=tf.get_variable('g_b1',shape=[4*4*1024],initializer=tf.contrib.layers.xavier_initializer())
        g_a1=tf.matmul(ip,g_w1)+g_b1
        g_a1=tf.reshape(g_a1,[-1,4,4,1024])
        g_a1=tf.layers.batch_normalization(g_a1,training=training_flag,reuse=reuse_flag)
        g_a1=tf.maximum(g_a1,beta*g_a1)

        g_w2=tf.get_variable('g_w2',shape=[3,3,512,1024],initializer=tf.contrib.layers.xavier_initializer())
        g_b2=tf.get_variable('g_b2',shape=[512],initializer=tf.contrib.layers.xavier_initializer())
        g_a2=tf.nn.conv2d_transpose(g_a1,g_w2,output_shape=[batch_size,8,8,512],strides=[1,2,2,1],padding='SAME')+g_b2
        g_a2=tf.layers.batch_normalization(g_a2,training=training_flag,scale=True,reuse=reuse_flag)
        g_a2=tf.maximum(g_a2,beta*g_a2)

        g_w3=tf.get_variable('g_w3',shape=[3,3,256,512],initializer=tf.contrib.layers.xavier_initializer())
        g_b3=tf.get_variable('g_b3',shape=[256],initializer=tf.contrib.layers.xavier_initializer())
        g_a3=tf.nn.conv2d_transpose(g_a2,g_w3,output_shape=[batch_size,16,16,256],strides=[1,2,2,1],padding='SAME')+g_b3
        g_a3=tf.layers.batch_normalization(g_a3,training=training_flag,scale=True,reuse=reuse_flag)
        g_a3=tf.maximum(g_a3,beta*g_a3)

        g_w4=tf.get_variable('g_w4',shape=[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer())
        g_b4=tf.get_variable('g_b4',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_a4=tf.nn.conv2d_transpose(g_a3,g_w4,output_shape=[batch_size,32,32,128],strides=[1,2,2,1],padding='SAME')+g_b4
        g_a4=tf.layers.batch_normalization(g_a4,training=training_flag,scale=True,reuse=reuse_flag)
        g_a4=tf.maximum(g_a4,beta*g_a4)

        g_w5=tf.get_variable('g_w5',shape=[3,3,3,128],initializer=tf.contrib.layers.xavier_initializer())
        g_b5=tf.get_variable('g_b5',shape=[3],initializer=tf.contrib.layers.xavier_initializer())
        g_a5=tf.nn.conv2d_transpose(g_a4,g_w5,output_shape=[batch_size,64,64,3],strides=[1,2,2,1],padding='SAME')+g_b5

        return tf.nn.sigmoid(g_a5),g_mu_a_0,g_sd_a_0


def show_image():

    encoded_sentences=pickle.load(open('./encoded_vector/test_vector.pkl','rb'))
    print(np.shape(encoded_sentences))
    gen_images=sess.run(generated_images_tensor,feed_dict={encoded_sentences_tensor:encoded_sentences[-1].reshape(1,4800),training_flag:False})
    print(np.shape(gen_images))

    return gen_images[0]


if __name__ == "__main__":
    _captions = []
    encoded_sentences_tensor=tf.placeholder(tf.float32,shape=[batch_size,4800])
    training_flag = tf.placeholder(tf.bool) 


    generated_images_tensor,_,_=Generator(batch_size,noise_length,encoded_sentences_tensor,False,training_flag)
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()

    saver.restore(sess,'./saved_models/c2i-255000')
    print("Model restored.")
    while True:

        a = input("Text: ")
        _captions.append(a)
        encoding(_captions)
        gen_images = []
        fig=plt.figure(figsize=(8, 8))
        columns = 2
        rows = 3
        for i in range(1,columns*rows +1):
            img = show_image()
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()



