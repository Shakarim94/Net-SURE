from __future__ import print_function
from __future__ import division
import os
import numpy as np
import argparse

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
 
from model import denoiser

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200, help='minibatch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--sigma', dest='sigma', type=float, default=50.0, help='noise level')
parser.add_argument('--eps', dest='eps', type=float, default=0.0001, help='epsilon') #see eq. (6) on the paper
parser.add_argument('--cost', dest='cost', default='sure', help='cost to minimize (sure or mse)')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='samples are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test samples are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='tensorboard logs are saved here')

parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--gpu', dest='gpu', default='0', help='which gpu to use')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--type', dest='type', default='', help='optional string, adds unique names to folders and files')
args = parser.parse_args()


def denoiser_train(model, lr):
    model.train(batch_size=args.batch_size, epoch=args.epoch, lr=lr)

def denoiser_test(model, save_dir):
    model.test(save_dir)


def main(_):
    
    #the following string is attached to checkpoint, log and image folder names
    eps_str = str(int(np.log10(args.eps)))
    name = "NET_" + args.cost + "_eps1e" + eps_str + "_sigma" + str(int(args.sigma)) + "_" + str(args.type)

    ckpt_dir = args.ckpt_dir + "/" + name
    sample_dir = args.sample_dir + "/" + name
    test_dir = args.test_dir + "/" + name
    log_dir = args.log_dir + "/" + name
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    #learning rate decay schedule
    lr = args.lr * np.ones([args.epoch])
    lr[100:] = lr[0] / 10.0 #lr decay after 100 epochs
    
    #dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    print("Model: %s" % (name))
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, dataset=mnist, sigma=args.sigma, eps=args.eps, cost_str=args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, dataset=mnist, sigma=args.sigma, eps=args.eps, cost_str=args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    tf.app.run()
