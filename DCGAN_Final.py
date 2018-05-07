import tensorflow as tf
import numpy as np
import os
import pickle
import random
from matplotlib import pyplot as plt


beta=0.3 #Leak Factor
keep_prob=0.7 #Dropout layer parameter
noise_length=100 #Noise vector length
batch_size=128 # Batch Size for training
dataset=open('/home/SharedData/yagnesh/CUB/list.txt','r').read().splitlines() #List of all images
epoch=1 #current training iteration
epochs=255000 #upper limit to the iterations
lam=2.0 #divergence loss lagrange multiplier

#Function to generate a batch of images and corresponding encoded sentences tensor
def generate_cub_batch(batch_size):
    batch=random.sample(dataset,batch_size) #Sample from images at random
    real_images=np.empty(shape=[batch_size,64,64,3]) #For reading groundtruth images
    encoded_sentence=np.empty(shape=[batch_size,4800]) #For reading encoded sentences
    for i in range(len(batch)):
        real_images[i]=plt.imread('/home/SharedData/yagnesh/CUB/images_64/'+batch[i]+'.png')
        sentence_list=np.load('/home/SharedData/yagnesh/CUB/encoded_captions/'+batch[i]+'.npy')
        encoded_sentence[i]=sentence_list[random.randint(0,np.shape(sentence_list)[0]-1)]
    return real_images,encoded_sentence

#Setting up GPU parameters
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess=tf.Session(config=config)


#Generator Network
def Generator(batch_size,z_len,encoded_sentences_tensor,reuse_flag,training_flag):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_flag):

        #Learning mean and variance of the latent conditioning variable
        g_mu_w_0=tf.get_variable('g_mu_w_0',shape=[4800,128],initializer=tf.contrib.layers.xavier_initializer())
        g_mu_b_0=tf.get_variable('g_mu_b_0',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_mu_a_0=tf.matmul(encoded_sentences_tensor,g_mu_w_0)+g_mu_b_0
        g_mu_a_0=tf.maximum(g_mu_a_0,beta*g_mu_a_0)

        g_sd_w_0=tf.get_variable('g_sd_w_0',shape=[4800,128],initializer=tf.contrib.layers.xavier_initializer())
        g_sd_b_0=tf.get_variable('g_sd_b_0',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        g_sd_a_0=tf.matmul(encoded_sentences_tensor,g_sd_w_0)+g_sd_b_0
        g_sd_a_0=tf.maximum(g_sd_a_0,beta*g_sd_a_0)

        #Sampling from latent conditioning variable
        epsilon=tf.random_normal([batch_size,128],mean=0,stddev=1)
        g_cond=g_mu_a_0+tf.multiply(g_sd_a_0,epsilon)

        #Sampling from latent noise variable
        z=tf.random_normal([batch_size,z_len],mean=0,stddev=1)
        ip=tf.concat([z,g_cond],1)

        #Decoder network
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


#Discriminator Network
def Discriminator(image_tensor,encoded_sentences_tensor,reuse_flag):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_flag):

        #Caption encoder
        d_w0=tf.get_variable('d_w0',shape=[4800,4*4*128],initializer=tf.contrib.layers.xavier_initializer())
        d_b0=tf.get_variable('d_b0',shape=[4*4*128],initializer=tf.contrib.layers.xavier_initializer())
        d_a0=tf.matmul(encoded_sentences_tensor,d_w0)+d_b0
        d_a0=tf.maximum(d_a0,beta*d_a0)
        d_a0=tf.reshape(d_a0,shape=[batch_size,4,4,128])

        #Encoder network
        d_w1=tf.get_variable('d_w1',shape=[3,3,3,64],initializer=tf.contrib.layers.xavier_initializer())
        d_b1=tf.get_variable('d_b1',shape=[64],initializer=tf.contrib.layers.xavier_initializer())
        d_a1=tf.nn.conv2d(image_tensor,d_w1,strides=[1,2,2,1],padding='SAME')+d_b1
        d_a1=tf.maximum(d_a1,beta*d_a1)
        d_a1=tf.nn.dropout(d_a1,keep_prob)

        d_w2=tf.get_variable('d_w2',shape=[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())
        d_b2=tf.get_variable('d_b2',shape=[128],initializer=tf.contrib.layers.xavier_initializer())
        d_a2=tf.nn.conv2d(d_a1,d_w2,strides=[1,2,2,1],padding='SAME')+d_b2
        d_a2=tf.maximum(d_a2,beta*d_a2)
        d_a2=tf.nn.dropout(d_a2,keep_prob)

        d_w3=tf.get_variable('d_w3',shape=[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer())
        d_b3=tf.get_variable('d_b3',shape=[256],initializer=tf.contrib.layers.xavier_initializer())
        d_a3=tf.nn.conv2d(d_a2,d_w3,strides=[1,2,2,1],padding='SAME')+d_b3
        d_a3=tf.maximum(d_a3,beta*d_a3)
        d_a3=tf.nn.dropout(d_a3,keep_prob)

        d_w4=tf.get_variable('d_w4',shape=[3,3,256,512],initializer=tf.contrib.layers.xavier_initializer())
        d_b4=tf.get_variable('d_b4',shape=[512],initializer=tf.contrib.layers.xavier_initializer())
        d_a4=tf.nn.conv2d(d_a3,d_w4,strides=[1,2,2,1],padding='SAME')+d_b4
        d_a4=tf.maximum(d_a4,beta*d_a4)
        d_a4=tf.nn.dropout(d_a4,keep_prob)

        #Augmenting condition
        d_a4=tf.concat([d_a4,d_a0],axis=3)

        d_w5=tf.get_variable('d_w5',shape=[4,4,512+128,1024],initializer=tf.contrib.layers.xavier_initializer())
        d_b5=tf.get_variable('d_b5',shape=[1024],initializer=tf.contrib.layers.xavier_initializer())
        d_a5=tf.nn.conv2d(d_a4,d_w5,strides=[1,1,1,1],padding='VALID')+d_b5
        d_a5=tf.maximum(d_a5,beta*d_a5)
        d_a5=tf.nn.dropout(d_a5,keep_prob)
        d_a5=tf.reshape(d_a5,[batch_size,1024])

        d_w6=tf.get_variable('d_w6',shape=[1024,1],initializer=tf.contrib.layers.xavier_initializer())
        d_b6=tf.get_variable('d_b6',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
        d_a6=tf.matmul(d_a5,d_w6)+d_b6

        return d_a6

#Defining placeholders
encoded_sentences_tensor=tf.placeholder(tf.float32,shape=[batch_size,4800])
real_images_tensor=tf.placeholder(tf.float32,shape=[batch_size,64,64,3])
arbit_sentences_tensor=tf.placeholder(tf.float32,shape=[batch_size,4800])
training_flag = tf.placeholder(tf.bool)

#Generating images from sentence embeddings
generated_images_tensor,mu1,sd1=Generator(batch_size,noise_length,encoded_sentences_tensor,False,training_flag)
#Discrminating generated images, real images with real descriptions and real images with unmatched descriptions
Dg=Discriminator(generated_images_tensor,encoded_sentences_tensor,False)
Dx=Discriminator(real_images_tensor,encoded_sentences_tensor,True)
Db=Discriminator(real_images_tensor,arbit_sentences_tensor,True)

#Cross entropy loss for generator
W_g_ce=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.ones_like(Dg)-0.1))
#KLD loss for conditioning latent variable
N_1=tf.distributions.Normal(mu1,sd1)
N_1_n=tf.distributions.Normal(tf.zeros([1,128]),tf.ones([1,128]))
W_g_kl=tf.reduce_mean(tf.contrib.distributions.kl_divergence(N_1,N_1_n))
W_g=W_g_ce+lam*W_g_kl

#Cross entropy loss for discriminator
W_d_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.ones_like(Dx)-0.1))
W_d_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.zeros_like(Dg)+0.1))
W_d_bad=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Df,labels=tf.zeros_like(Db)+0.1))
W_d=W_d_real+W_d_fake+W_d_bad

#Segregating generator and discriminator variables
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]
var_names=[var.name for var in tf.trainable_variables()]

#Defining optimizers
update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_trainer=tf.train.AdamOptimizer(25e-6,beta1=0.5).minimize(W_d,var_list=d_vars)
    g_trainer=tf.train.AdamOptimizer(25e-6,beta1=0.5).minimize(W_g, var_list=g_vars)

#Initialize networks and define saver
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
# saver.restore(sess,'/home/SharedData/yagnesh/c2i/saved_model_stage1/c2i-'+str(epoch))

#Training code
while epoch<=epochs:
    #Sampling mini batch
    real_images,encoded_sentences=generate_cub_batch(batch_size)
    _,arbit_sentences=generate_cub_batch(batch_size)

    #Discriminator update
    _,dLoss=sess.run([d_trainer,W_d],feed_dict={real_images_tensor:real_images,
                                                encoded_sentences_tensor:encoded_sentences,
                                                arbit_sentences_tensor:arbit_sentences,
                                                training_flag:True})
    #Sampling mini batch
    real_images,encoded_sentences=generate_cub_batch(batch_size)
    #Generator update
    _,gLoss=sess.run([g_trainer,W_g],feed_dict={real_images_tensor:real_images,
                                                encoded_sentences_tensor:encoded_sentences,
                                                training_flag:True})

    print(epoch,gLoss,dLoss)

    #Saving model and images
    if epoch%100==0:
        saver.save(sess,'/home/SharedData/yagnesh/c2i/saved_model_stage1/c2i',global_step=epoch)
        gen_images=sess.run(generated_images_tensor,feed_dict={real_images_tensor:real_images,
                                                               encoded_sentences_tensor:encoded_sentences,
                                                               training_flag:False})
        plt.imsave(arr=gen_images[0],fname='/home/SharedData/yagnesh/c2i/outputs_stage1/'+str(epoch)+'_gen.png')
        plt.imsave(arr=real_images[0],fname='/home/SharedData/yagnesh/c2i/outputs_stage1/'+str(epoch)+'_gt.png')
    epoch=epoch+1
