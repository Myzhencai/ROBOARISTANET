#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 This is the implemention of the paper:Deep Multimodal Embedding: Manipulating Novel Objects with Point-clouds, Language, Trajectories
Jaeyong Sung, Ian Lenz, Ashutosh Saxena
 This is the network framework
'''


#导入网络架构所需的包
import numpy as np
import tensorflow as tf
from loss_h2_pl import Loss_h2_pl 
from loss_h3 import Loss_h3
from loss_tr_duplicated import Loss_tr_duplicated


#我后期会改成hdrnet中参数的初始化方式并且需要对最后几层的参数需要导入预训练参数

def xavier_init(fan_in,fan_out,constant = 1):
    low   = -constant * np.sqrt(6.0/(fan_in+fan_out))
    hight =  constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),maxval=hight,minval=low,dtype=tf.float32)


#这是 spare de-noising autoencoder(SDA)
class NeedParameter(object):
     def __init__(self,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha=0.2,alpha_t，alpha_r，beta，gama):
        super(Layer2_transition1,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1)
        self.n_hidden1 = n_hidden1
        self.pointcloud_i = pointcloud_i
        self.language_i = language_i
        self.most_fit_trajectory_star = most_fit_trajectory_star
        self.trajectory_Set_I = trajectory_Set_I
        self.language_Set = language_Set
        self.tS这是一个阀值 = tS这是一个阀值
        self.tD这是一个阀值 = tD这是一个阀值
        self.alpha = alpha
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self.beta = beta 
        self.gama = gama

class SDA(NeedParameter):
    def __init__(self,n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
       super(SDA,self).__init__(pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
        self.n_input = n_input #point-cloud/language unit number这需要设定一个函数来计算
        self.n_hidden = n_hidden #the first layer unit number
        self.activate = activate_function#the activation function is relu as the paper
        self.training_scale = scale #the degree of noise 
        self.weights = dict()#make weights the dictory
     
    #the network frame forspare de-noising autoencoder
        with tf.name_scope('Rawinput'):
            self.input = tf.placeholder(tf.float32,[None，self.n_input])#for the reason that we need to modify the input of the p/l so we need to change it !!! 
        with tf.name_scope('NoiseInput'):
            self.scale = tf.placeholder(dtype=tf.float32)
            self.noise_mask = np.random.binomial(1, 1 - self.scale, self.n_input) #use the binomial noise with corruption level p=0.1 as the 
            self.noise_input = tf.add(self.input,self.noise_mask)
        with tf.name_scope('Encoder'):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight')
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='bias')
            self.hidden = self.activate(tf.add(tf.matmul(self.noise_input,self.weights['w1']),self.weights['b1']))#this is the result of the first layer because we need to train it we need to reconstructed it and when we use the network we can only use the parameter for use (weights)    
        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_input), name='weight_encoder')
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='bias_encoder') 
            self.__reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),self.weights['b2'])
     #this is the loss and optimizer
        with tf.name_scope('Losslayer1'):
            self.__sqrt_hidden = tf.sqrt(tf.square(self.hidden))
            self.__modified_sqrt_hidden = 0.3* self.__sqrt_hidden #the value 0.3 should change cause the paper did not mention the value
            self.__reduce_sum = tf.reduce_sum(tf.pow(tf.sub(self.__reconstruction, self.input), 2))
            self.__loss =  self.__reduce_sum + self.__modified_sqrt_hidden#need to look the deeplearning book to understand the foram !!

        with tf.name_scope('Optimizerlayer1'):
            self.__optimizer = optimizer.minimize(self.__loss)

    def Caculateloss(self,Input):
        loss = self.sess.run(self.__loss,feed_dict={self.input:Input,self.scale:self.training_scale})
        return loss
    def Train(self,Input):
        train = self.sess.run(self.__optimizer,feed_dict={self.input:Input,self.scale:self.training_scale})

        '''
        每个网络的训练法子不一样
        '''
        


#这是神经网络第二层的过渡态

class Layer2_transition1(SDA):
     def __init__(self,n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
        super(Layer2_transition1,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
        self.n_hidden1 = n_hidden1
   
        with tf.name_scope('Hiddenlayer2part1'):
            self.weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_hidden1), name='weight_secondlayerpart1')#可以参考hdrnet对参数进行初始化
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_hidden1], dtype=tf.float32), name='bias_secondlayerpart1')
            self.hidden1_Transitionperiod = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
            

class Layer2_transition2(SDA):
     def __init__(self,n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
        super(Layer2_transition2,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
        self.n_hidden1 = n_hidden1
        
   
        with tf.name_scope('Hiddenlayer2part2'):
            self.weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_hidden1), name='weight_secondlayerpart2')#可以参考hdrnet对参数进行初始化
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_hidden1], dtype=tf.float32), name='bias_secondlayerpart2')
            self.hidden1_Transitionperiod = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

class Layer2_transition3(SDA):
     def __init__(self,n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
        super(Layer2_transition3,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
        self.n_hidden1 = n_hidden1
        
        with tf.name_scope('Hiddenlayer2part3'):
            self.weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_hidden1), name='weight_secondlayerpart2')#可以参考hdrnet对参数进行初始化
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_hidden1], dtype=tf.float32), name='bias_secondlayerpart2')
            self.hidden1_Transitionperiod = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

#这是神经网络第二层融合
class Layer2pointcloudlanguage(Layer2_transition1,Layer2_transition2,Layer2_transition3):

     def __init__(self,n_inputpart1,n_hiddenpart1,n_inputpart2,n_hiddenpart2,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):

        Layer2_transition1.__init__(self,n_inputpart1,n_hiddenpart2,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)#向第二层的第一个分量传递参数

        Layer2_transition2.__init__(self,n_inputpart2,n_hiddenpart2,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)#向第二层的第二个分量传递参数

        Layer2_transition3.__init__(self,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)#向第二层的第三个分量传递参数

        with tf.name_scope('MergeHiddenlayer2'):
            self.Layer2_part1 = Layer2_transition1.hidden1_Transitionperiod
            self.Layer2_part2 = Layer2_transition2.hidden1_Transitionperiod#可能参数会重合也可能通过保存在不同的路径中避免
            self.Layer2_part3 = Layer2_transition3.hidden1_Transitionperiod
            self.MergeLayer2_output = tf.activate(tf.add(self.Layer2_part1,self.Layer2_part2))
        #Layer2_output =tf.activate(tf.add(Layer2_transition(填写相应的参数这是云点的).hiddenlayer_output(),Layer2_transition(填写相应的参数这是语音的).hiddenlayer_output()))
        #return Layer2_output#第二隐藏层的输出h 2,pl
        with tf.name_scope('Losslayer2'):
            self.__losslayer2 = Loss_h2_pl(self.pointcloud_i,self.language_i,self.most_fit_trajectory_star,self.trajectory_Set_I,self.language_Set,self.tS这是一个阀值,self.tD这是一个阀值,self.alpha = alpha,self.alpha_t,self.alpha_r,self.beta,self.gama).loss_function_h2(number需要修改和咨询作者，self.Layer2_part1，self.Layer2_part2，self.Layer2_part3)#Layer2_transition(填写相应的参数这是云点的).hidden1_Transitionperiod , Layer2_transition(填写相应的参数这是最反常的语音的).hidden1_Transitionperiod ,Layer2_transition(填写相应的参数这是语音的).hidden1_Transitionperiod这是相应语言的
        #可能需要计算出相应的平均值
        with tf.name_scope('Optimizerlayer2'):
            self.__optimizer = optimizer.minimize(self.__losslayer2)

    def Caculateloss(self,还有其他参数):
        loss = self.sess.run(self.__losslayer2,feed_dict={其他参数})
        return loss
    def Train(self,还有其他参数):
        train = self.sess.run(self.__optimizer,feed_dict={其他参数})
 

class Layer2trajectory(Layer2pointcloudlanguage):
      def __init__(self,n_inputpart1,n_hiddenpart1,n_inputpart2,n_hiddenpart2,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
          super(Layer2trajectory,self).__init__(n_inputpart1,n_hiddenpart1,n_inputpart2,n_hiddenpart2,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
         with tf.name_scope('Losslayer2'):
              self.__losslayer2 = Loss_tr_duplicated(self.pointcloud_i,self.language_i,self.most_fit_trajectory_star,self.trajectory_Set_I,self.tS这是一个阀值,self.tD这是一个阀值,self.alpha = alpha,self.alpha_t,self.alpha_r,self.beta,self.gama).loss_function(number需要修改和咨询作者，self.Layer2_part1，self.Layer2_part2，self.Layer2_part3)#Layer2_transition(填写相应的参数这是云点的).hidden1_Transitionperiod , Layer2_transition(填写相应的参数这是最反常的语音的).hidden1_Transitionperiod ,Layer2_transition(填写相应的参数这是语音的).hidden1_Transitionperiod这是相应语言的
        #可能需要计算出相应的平均值 
         with tf.name_scope('Optimizerlayer2'):
              self.__optimizer = optimizer.minimize(self.__losslayer2)
 
     def Caculateloss(self,还有其他参数):
        loss = self.sess.run(self.__losslayer2,feed_dict={其他参数})
        return loss
     def Train(self,还有其他参数):
        train = self.sess.run(self.__optimizer,feed_dict={其他参数})
   

#这是神经网络第三层
class Layer3_pointcloudlanguage(Layer2pointcloudlanguage):
     def __init__(self,n_inputpart1,n_hiddenpart1,n_inputpart2,n_hiddenpart2,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,n_hidden2,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
        super(Layer2pointcloudlanguage,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
        self.n_hidden2 = n_hidden2#这是第三层即最后的输出层的神经元个数

        #这是第三层
        with tf.name_scope('Hiddenlayer3'):
            self.weights['w3'] = tf.Variable(xavier_init(self.n_hidden1, self.n_hidden2), name = 'weight_thirdlayer')
            self.weights['b3'] = tf.Variable(tf.zeros([self.n_hidden2],dtype=tf.float32),name = 'b_thirdlayer')
            self.hidden3 = tf.activate(tf.add(tf.matmul(self.MergeLayer2_output,self.weights['w3']),self.weithts['b3']))

class Layer3_trajectory(Layer2_transition1):
     def __init__(self,n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,n_hidden2,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
         super(Layer2_transition1,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
         self.n_hidden2 = n_hidden2
         with tf.name_scope('ActivateHiddenlayer2'):
             self.activatelayer2 = tf.activate(self.hidden1_Transitionperiod)
              
         with tf.name_scope('Hiddenlayer3'):
             self.weights['w3'] = tf.Variable(xavier_init(self.n_hidden1, self.n_hidden2), name = 'weight_thirdlayer')
             self.weights['b3'] = tf.Variable(tf.zeros([self.n_hidden2],dtype=tf.float32),name = 'b_thirdlayer')
             self.hidden3 = tf.activate(tf.add(tf.matmul(self.activatelayer2 ,self.weights['w3']),self.weithts['b3']))

class Layer3(Layer3_pointcloudlanguage,Layer3_trajectory):
     def __init__(self,n_inputpart1,n_hiddenpart1,n_inputpart2,n_hiddenpart2,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,n_hidden1_tr,n_hidden2,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama,n_input,n_hidden):#,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,n_hidden2,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama
     super(Layer3_pointcloudlanguage,self).__init__(n_inputpart1,n_hiddenpart1,n_inputpart2,n_hiddenpart2,n_inputpart3,n_hiddenpart3,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1,n_hidden2,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
     super(Layer3_trajectory,self).__init__(n_input,n_hidden,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1_tr,n_hidden2,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama):
       with tf.name_scope('Losslayer3'):
            self.__losslayer3 = Loss_h3(写上改好的参数).loss_function_h3(写上改好的参数)#可能损失函数需要修改

       with tf.name_scope('Optimizerlayer3'):
            self.__optimizer = optimizer.minimize(self.__losslayer3)
#至此网络架构的类形成可以直接从调用类直接生成一个相应的网络
    def Caculateloss(self,还有其他参数):
        loss = self.sess.run(self.__losslayer3,feed_dict={其他参数})
        return loss
    def Train(self,还有其他参数):
        train = self.sess.run(self.__optimizer,feed_dict={其他参数})
   

