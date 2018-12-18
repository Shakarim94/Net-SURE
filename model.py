from __future__ import print_function
from __future__ import division
import os
import numpy as np
import time
import tensorflow as tf

from utils import *
from scipy.misc import imsave as ims

#Denoising Autoencoder
def NET(input):
    xavier_init = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("Encoder_1"):
        out = tf.layers.conv2d(input, 10, 3, (2,2), padding='same', activation=tf.sigmoid, kernel_initializer=xavier_init)
   
    with tf.variable_scope("Encoder_2"):
        out = tf.layers.conv2d(out, 10, 3, (2,2), padding='same', activation=tf.sigmoid, kernel_initializer=xavier_init)

    with tf.variable_scope("Decoder_2"):
        out = tf.layers.conv2d_transpose(out, 10, 3, (2,2), padding='same', activation=tf.sigmoid, kernel_initializer=xavier_init)
    
    with tf.variable_scope("Decoder_1"):
        out = tf.layers.conv2d_transpose(out, 1, 3, (2,2), padding='same', activation=tf.sigmoid, kernel_initializer=xavier_init)

    return out


class denoiser(object):
    def __init__(self, sess, dataset, sigma, eps, cost_str, ckpt_dir, sample_dir, log_dir):
        self.sess = sess
        self.sigma = sigma
        self.eps = eps
        self.cost_str = cost_str
        
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        
        
        #datasets
        self.dataset = dataset
        self.train_set = self.dataset.train.images
        self.valid_set = self.dataset.validation.images
        
        x_dim = int(np.sqrt(np.shape(self.train_set)[1]))
        
        self.train_set = np.reshape(self.train_set, [-1, x_dim, x_dim, 1])
        self.valid_set = np.reshape(self.valid_set, [-1, x_dim, x_dim, 1])
        
        # build model
        #placeholders for clean and noisy image batches
        self.X = tf.placeholder(tf.float32, [None, x_dim, x_dim, 1], name='clean_image')
        self.Y = tf.placeholder(tf.float32, [None, x_dim, x_dim, 1], name='noisy_image')
        #self.Y = self.X + tf.random_normal(shape=tf.shape(self.X), stddev=self.sigma / 255.0)  # noisy images
        
        
        #forward propagation
        with tf.variable_scope('NET'):
            self.Y_ = NET(self.Y)
        
        #forward propagation of the perturbed input
        self.n = tf.random_normal(shape=tf.shape(self.Y), stddev=1.0) #see eq (6) on the paper
        self.Z = self.Y + self.n*self.eps
        with tf.variable_scope('NET', reuse=True):
            self.Z_ = NET(self.Z)
    
        self.var = (self.sigma/255.0)**2
        batch = tf.to_float(tf.shape(self.Y)[0]) #size of the minibatch
        
        #COST FUNCTIONS
        self.mse = (2.0 / batch) * tf.nn.l2_loss(self.Y_ - self.X)
        
        self.divergence = (1.0/self.eps)*(tf.reduce_sum(tf.multiply(self.n, (self.Z_-self.Y_))))

        #eq. (14)
        self.sure = (1.0 / batch)*(2.0*tf.nn.l2_loss(self.Y - self.Y_) - batch*x_dim*x_dim*self.var + 2.0*self.var*self.divergence)
        
        #which cost function to use for training
        if self.cost_str=='sure':
            self.cost = self.sure
        elif self.cost_str=='mse':
            self.cost = self.mse
        else:
            print("UNKNOWN COST")
        
        
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eval_psnr = tf.placeholder(tf.float32, name='eval_psnr') #avg. PSNR on validation set
        
        #optimizer
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_op = optimizer.minimize(self.cost)
        
        #enable when using batchnorm
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
            #self.train_op = optimizer.minimize(self.cost)
        
        #checkpoint saver
        self.saver = tf.train.Saver(max_to_keep=5)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialized model successfully...")

    #function to evaluate on validation set
    def evaluate(self, valid_corrupt, iter_num, summary_merged, summary_writer):

        feed_dict_eval={self.X : self.valid_set, self.Y: valid_corrupt}
        valid_output, valid_mse, valid_sure = self.sess.run([self.Y_, self.mse, self.sure],feed_dict=feed_dict_eval)
        
        groundtruth = np.clip(255 * np.squeeze(self.valid_set), 0, 255).astype('uint8')
        noisyimage = np.clip(255 * np.squeeze(valid_corrupt), 0, 255).astype('uint8')
        outputimage = np.clip(255 * np.squeeze(valid_output), 0, 255).astype('uint8')
        # calculate PSNR
        avg_psnr = cal_psnr(groundtruth, outputimage)
        psnr_summary = self.sess.run(summary_merged, feed_dict={self.eval_psnr:avg_psnr})
        summary_writer.add_summary(psnr_summary, iter_num)
        
        #saving sample images
        ims(self.sample_dir+"/"+str(iter_num)+".png", merge(outputimage[:100], [10,10]))
        
        
        print("VALID, mse: %.6f, sure: %.6f" % (valid_mse, valid_sure))
        print("---- Validation Set ---- Average PSNR %.3f ---" % avg_psnr)

    def train(self, batch_size, epoch, lr):
        numBatch = int(self.dataset.train.num_examples / batch_size)
        
        # load pretrained model
        load_model_status, global_step = self.load(self.ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = (global_step) // numBatch
            start_step = (global_step) % numBatch
            print("[*] Model restored successfully!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Pretrained model not found!")
        # make summary
        tf.summary.scalar('mse', self.mse)
        tf.summary.scalar('sure', self.sure)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter(self.log_dir+"/", self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eval_psnr', self.eval_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        
        #generating corrupted images (only once)
        valid_noisy = self.valid_set + np.random.normal(0, self.sigma/255.0, np.shape(self.valid_set)).astype('float32')
        train_noisy = self.train_set + np.random.normal(0, self.sigma/255.0, np.shape(self.train_set)).astype('float32')

        ims(self.sample_dir+"/valid_groundtruth.png", merge(np.squeeze(self.valid_set[:100]), [10,10]))
        ims(self.sample_dir+"/valid_noisy.png", merge(np.squeeze(valid_noisy[:100]), [10,10]))

        self.evaluate(valid_noisy, iter_num, summary_merged=summary_psnr, summary_writer=writer)
        
        tf.get_default_graph().finalize() #making sure that the graph is fixed at this point
        
        print("Training set shape:")
        print(np.shape(train_noisy))
        print("\n")
        #training loop
        for epoch in xrange(start_epoch, epoch):
            
            #shuffling two arrays in unison
            rng_state = np.random.get_state()
            np.random.shuffle(self.train_set)
            np.random.set_state(rng_state)
            np.random.shuffle(train_noisy)
            
            for batch_id in xrange(0, numBatch):
                
                batch_images = self.train_set[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_images_corrupt = train_noisy[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                feed_dict = {self.X: batch_images, self.Y: batch_images_corrupt, self.lr: lr[epoch]}
                
                self.sess.run(self.train_op, feed_dict=feed_dict)
                
                #save summaries
                if (iter_num)%25==0:
                    summary = self.sess.run(merged,feed_dict=feed_dict)
                    writer.add_summary(summary, iter_num)
                
                iter_num += 1

            mse, sure = self.sess.run([self.mse, self.sure],feed_dict=feed_dict)

            print("Minimizing {}".format(self.cost_str))
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time))
            print("TRAIN, mse: %.6f, sure: %.6f" % (mse, sure))
            
            #evalute on the validation set
            self.evaluate(valid_noisy, iter_num, summary_merged=summary_psnr,
                                summary_writer=writer)

            #saving checkpoint
            self.save(iter_num, self.ckpt_dir)
            print("\n")
                
            
        print("[*] Finish training.")


    #checkpoint saverfunction
    def save(self, iter_num, checkpoint_dir, model_name='NET-tensorflow'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)
    
    #checkpoint loader function
    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, save_dir):

        # init variables
        tf.global_variables_initializer().run()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        load_model_status, global_step = self.load(self.ckpt_dir)
        assert load_model_status == True, '[!] Loading weights FAILED...'
        print(" [*] Loading weights SUCCESS...")
        
        test_set = np.reshape(self.dataset.test.images, [-1, 28, 28, 1])
        test_corrupt = test_set + np.random.normal(0, self.sigma/255.0, np.shape(test_set)).astype('float32')
        
        feed_dict_test={self.Y: test_corrupt}
        recon = self.sess.run(self.Y_, feed_dict=feed_dict_test)
        
        groundtruth = np.clip(255 * np.squeeze(test_set), 0, 255).astype('uint8')
        outputimage = np.clip(255 * np.squeeze(recon), 0, 255).astype('uint8')
        # calculate PSNR
        avg_psnr_test = cal_psnr(groundtruth, outputimage)
        
        #saving sample images
        ims(save_dir+"/denoised.png", merge(outputimage[:100], [10,10]))
        ims(save_dir+"/noisy.png", merge(np.squeeze(test_corrupt)[:100], [10,10]))

        print("---- TEST SET ---- Average PSNR %.3f ---" % avg_psnr_test)
