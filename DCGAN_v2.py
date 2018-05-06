import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

beta_g=0.2
beta=0.2
drop_rate=0.4
noise_length=100
batch_size=2
annotation_dir = 'coco/test_annotations'


train_images= 'train_images/'
encoded_vector_dir = 'captions_encoded/'

epoch=64501
epochs=1000000

def generate_coco_batch(batch_size):
    batch=random.sample(listdir(train_images),batch_size)
    real_images=np.empty(shape=[batch_size,64,64,3])
    encoded_sentence=np.empty(shape=[batch_size,4800])
    for i in range(len(batch)):
        try:
            real_images[i]=plt.imread(train_images+batch[i])
            sentence_list=pickle.load(open(encoded_vector_dir+batch[i].strip('.png')+'.pkl','rb'))
            encoded_sentence[i]=sentence_list[random.randint(0,4)]
        except ValueError:
             print("Black and white image")

    return real_images,encoded_sentence


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess=tf.Session(config=config)


def Generator(batch_size,z_len,encoded_sentences_tensor,reuse_flag):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_flag):

        g_w0=tf.get_variable('g_w0',shape=[4800,128],initializer=tf.contrib.layers.xavier_initializer())
        g_b0=tf.get_variable('g_b0',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_a0=tf.matmul(encoded_sentences_tensor,g_w0)+g_b0
        g_a0=tf.tanh(g_a0)

        z=tf.truncated_normal([batch_size,z_len],mean=0,stddev=0.1) # Needs to be checked
        ip=tf.concat([z,g_a0],1)

        g_w1=tf.get_variable('g_w1',shape=[z_len+128,4*4*1024],initializer=tf.contrib.layers.xavier_initializer())
        g_b1=tf.get_variable('g_b1',shape=[4*4*1024],initializer=tf.contrib.layers.xavier_initializer())
        g_a1=tf.matmul(ip,g_w1)+g_b1
        g_a1=tf.reshape(g_a1,[-1,4,4,1024])
        g_a1=tf.maximum(g_a1,beta_g*g_a1)

        g_w2=tf.get_variable('g_w2',shape=[5,5,512,1024],initializer=tf.contrib.layers.xavier_initializer())
        g_b2=tf.get_variable('g_b2',shape=[512],initializer=tf.contrib.layers.xavier_initializer())
        g_a2=tf.nn.conv2d_transpose(g_a1,g_w2,output_shape=[batch_size,8,8,512],strides=[1,2,2,1],padding='SAME')+g_b2
        g_a2=tf.maximum(g_a2,beta_g*g_a2)

        g_w3=tf.get_variable('g_w3',shape=[5,5,256,512],initializer=tf.contrib.layers.xavier_initializer())
        g_b3=tf.get_variable('g_b3',shape=[256],initializer=tf.contrib.layers.xavier_initializer())
        g_a3=tf.nn.conv2d_transpose(g_a2,g_w3,output_shape=[batch_size,16,16,256],strides=[1,2,2,1],padding='SAME')+g_b3
        g_a3=tf.maximum(g_a3,beta_g*g_a3)

        g_w4=tf.get_variable('g_w4',shape=[5,5,128,256],initializer=tf.contrib.layers.xavier_initializer())
        g_b4=tf.get_variable('g_b4',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_a4=tf.nn.conv2d_transpose(g_a3,g_w4,output_shape=[batch_size,32,32,128],strides=[1,2,2,1],padding='SAME')+g_b4
        g_a4=tf.maximum(g_a4,beta_g*g_a4)
        

        g_w5=tf.get_variable('g_w5',shape=[5,5,64,128],initializer=tf.contrib.layers.xavier_initializer())
        g_b5=tf.get_variable('g_b5',shape=[64],initializer=tf.contrib.layers.xavier_initializer())
        g_a5=tf.nn.conv2d_transpose(g_a4,g_w5,output_shape=[batch_size,64,64,64],strides=[1,2,2,1],padding='SAME')+g_b5
        g_a5=tf.maximum(g_a5,beta*g_a5)
        
        
        g_w6=tf.get_variable('g_w6',shape=[5,5,3,64],initializer=tf.contrib.layers.xavier_initializer())
        g_b6=tf.get_variable('g_b6',shape=[3],initializer=tf.contrib.layers.xavier_initializer())
        g_a6=tf.nn.conv2d_transpose(g_a5,g_w6,output_shape=[batch_size,64,64,3],strides=[1,1,1,1],padding='SAME')+g_b6

        return tf.nn.sigmoid(g_a6)

def Discriminator(image_tensor,encoded_sentences_tensor,reuse_flag):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_flag):

        d_w0=tf.get_variable('d_w0',shape=[4800,64*64],initializer=tf.contrib.layers.xavier_initializer())
        d_b0=tf.get_variable('d_b0',shape=[64*64],initializer=tf.contrib.layers.xavier_initializer())
        d_a0=tf.matmul(encoded_sentences_tensor,d_w0)+d_b0
        d_a0=tf.tanh(d_a0)
        d_a0=tf.reshape(d_a0,shape=[-1,64,64,1])
        d_a0=tf.concat([image_tensor,d_a0],axis=3)

        d_w1=tf.get_variable('d_w1',shape=[5,5,4,64],initializer=tf.contrib.layers.xavier_initializer())
        d_b1=tf.get_variable('d_b1',shape=[64],initializer=tf.contrib.layers.xavier_initializer())
        d_a1=tf.nn.conv2d(d_a0,d_w1,strides=[1,2,2,1],padding='SAME')+d_b1        
        d_a1=tf.maximum(d_a1,beta*d_a1)
        d_a1=tf.nn.dropout(d_a1,drop_rate)
        #d_a1=tf.nn.relu(d_a1)

        d_w2=tf.get_variable('d_w2',shape=[5,5,64,128],initializer=tf.contrib.layers.xavier_initializer())
        d_b2=tf.get_variable('d_b2',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        d_a2=tf.nn.conv2d(d_a1,d_w2,strides=[1,2,2,1],padding='SAME')+d_b2
        d_a1=tf.maximum(d_a2,beta*d_a2)
        d_a2=tf.nn.dropout(d_a2,drop_rate)
        #d_a2=tf.nn.relu(d_a2)

        d_w3=tf.get_variable('d_w3',shape=[5,5,128,256],initializer=tf.contrib.layers.xavier_initializer())
        d_b3=tf.get_variable('d_b3',shape=[256],initializer=tf.contrib.layers.xavier_initializer())
        d_a3=tf.nn.conv2d(d_a2,d_w3,strides=[1,2,2,1],padding='SAME')+d_b3
        d_a3=tf.maximum(d_a3,beta*d_a3)
        d_a3=tf.nn.dropout(d_a3,drop_rate)
        #d_a3=tf.nn.relu(d_a3)

        d_w4=tf.get_variable('d_w4',shape=[5,5,256,512],initializer=tf.contrib.layers.xavier_initializer())
        d_b4=tf.get_variable('d_b4',shape=[512],initializer=tf.contrib.layers.xavier_initializer())
        d_a4=tf.nn.conv2d(d_a3,d_w4,strides=[1,2,2,1],padding='SAME')+d_b4
        d_a4=tf.maximum(d_a4,beta*d_a4)
        d_a4=tf.nn.dropout(d_a4,drop_rate)
        #d_a4=tf.nn.relu(d_a4)

        d_w5=tf.get_variable('d_w5',shape=[5,5,512,1024],initializer=tf.contrib.layers.xavier_initializer())
        d_b5=tf.get_variable('d_b5',shape=[1024],initializer=tf.contrib.layers.xavier_initializer())
        d_a5=tf.nn.conv2d(d_a4,d_w5,strides=[1,2,2,1],padding='SAME')+d_b5
        d_a5=tf.maximum(d_a5,beta*d_a5)
        d_a5=tf.nn.dropout(d_a5,drop_rate)
        #d_a5=tf.nn.relu(d_a5)
        #d_a5=tf.reshape(d_a5,[-1,2*2*1024])
        
        '''
        d_w6=tf.get_variable('d_w6',shape=[5,5,1024,2048],initializer=tf.contrib.layers.xavier_initializer())
        d_b6=tf.get_variable('d_b6',shape=[2048],initializer=tf.contrib.layers.xavier_initializer())
        d_a6=tf.nn.conv2d(d_a5,d_w6,strides=[1,2,2,1],padding='SAME')+d_b6
        d_a6=tf.maximum(d_a6,beta*d_a6)
        d_a6=tf.nn.dropout(d_a6,drop_rate)
        d_a6=tf.reshape(d_a6,[-1,2*2*2048])
        
        
        d_w6=tf.get_variable('d_w6',shape=[2*2*1024,1],initializer=tf.contrib.layers.xavier_initializer())
        d_b6=tf.get_variable('d_b6',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
        d_a6=tf.matmul(d_a5,d_w6)+d_b6
        d_a6=tf.nn.sigmoid(d_a6)
        '''

        return d_a5

encoded_sentences_tensor=tf.placeholder(tf.float32,shape=[batch_size,4800])
real_images_tensor=tf.placeholder(tf.float32,shape=[batch_size,64,64,3])
arbit_sentences_tensor=tf.placeholder(tf.float32,shape=[batch_size,4800])

generated_images_tensor=Generator(batch_size,noise_length,encoded_sentences_tensor,False)
Dg=Discriminator(generated_images_tensor,encoded_sentences_tensor,False)
Dx=Discriminator(real_images_tensor,encoded_sentences_tensor,True)
Df=Discriminator(real_images_tensor,arbit_sentences_tensor,True)
'''
W_g=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.ones_like(Dg)-0.1))
W_d_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.ones_like(Dx)-0.1))
W_d_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.zeros_like(Dg)+0.0))
W_d_bad=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Df,labels=tf.zeros_like(Df)+0.0))
W_d=W_d_real+W_d_fake+W_d_bad
'''
W_g=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Dg,pos_weight=2,targets=tf.ones_like(Dg)-0.1))
W_d_real=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Dx,pos_weight=2,targets=tf.ones_like(Dx)-0.1))
W_d_fake=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Dg,pos_weight=0.5,targets=tf.zeros_like(Dg)+0.0))
W_d_bad=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Df,pos_weight=0.5,targets=tf.zeros_like(Df)+0.0))
W_d=W_d_real+W_d_fake+W_d_bad




tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

d_trainer=tf.train.AdamOptimizer(0.5e-4,beta1=0.1).minimize(W_d,var_list=d_vars)
g_trainer=tf.train.AdamOptimizer(0.1e-4,beta1=0.1).minimize(W_g, var_list=g_vars)

sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()


saver.restore(sess,'./saved_models-soumya/c2i-43500')



while epoch<epochs:
    real_images,encoded_sentences=generate_coco_batch(batch_size)
    _,arbit_sentences=generate_coco_batch(batch_size)
    _,dLoss=sess.run([d_trainer,W_d],feed_dict={real_images_tensor:real_images, encoded_sentences_tensor:encoded_sentences,arbit_sentences_tensor:arbit_sentences})   
    real_images,encoded_sentences=generate_coco_batch(batch_size)
    _,gLoss=sess.run([g_trainer,W_g],feed_dict={real_images_tensor:real_images, encoded_sentences_tensor:encoded_sentences})


    if epoch%100==0:
        saver.save(sess,'./saved_models-soumya/c2i',global_step=epoch)
        gen_images=sess.run(generated_images_tensor,feed_dict={real_images_tensor:real_images, encoded_sentences_tensor:encoded_sentences})
        plt.imsave(arr=gen_images[0],fname='./outputs-soumya/'+str(epoch)+'_gen.png')
        plt.imsave(arr=real_images[0],fname='./outputs-soumya/'+str(epoch)+'_gt.png')

    epoch=epoch+1
    print("epochs: "+ str(epoch)+" dloss: "+str(dLoss)+ "  gloss: "+str(gLoss))



