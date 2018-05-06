#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
1.搭建一个可以生成一层网络的训练并保存参数a
2.搭建一个可以生成两层网络的训练并保存参数b（调用a参数作为起始第一层的参数）
3.搭建一个可以生成两层网络的训练并保存参数c（调用b参数作为起始第一层的参数）
'''

from roboaristanet import *

#运用上面定义的类来构建我们所需的神经网络
'''
Pre-training lower layers
'''
Point_cloud_encoder = SDA(n_input=云点的数目还需看论文需要对数据进行预处理，n_hidden=250,activate_function=tf.nn.relu,optimzer=tf.train.AdadeltaOptimizer(learning_rate=0.01),scale=0.01)#initialize the network with the parameters

Language_encoder = SDA(n_input=语音的数目还需看论文需要对数据进行预处理，n_hidden=150,activate_function=tf.nn.relu,optimzer=tf.train.AdadeltaOptimizer(learning_rate=0.01),scale=0.01)#initialize the network with the parameters

Trajectory_encoder = SDA(n_input=轨迹数目还需看论文需要对数据进行预处理，n_hidden=100,activate_function=tf.nn.relu,optimzer=tf.train.AdadeltaOptimizer(learning_rate=0.01),scale=0.01)#initialize the network with the parameters也许还需要另外一个轨迹的与之对应

#还需要训练出参数需要继续写代码



'''      
Pre-training Joint Point-cloud/Language Model     
'''

Pointcloud_Language_Model = Layer2pointcloudlanguage(n_inputpart1=云点的数目还需看论文需要对数据进行预处理,n_hiddenpart1=250,n_inputpart2=语音的数目还需要看论文对数据进行处理,n_hiddenpart2=150,n_inputpart3=最反常语音的数目还需要看论文对数据进行处理,n_hiddenpart3=150,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1=125,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)#需要修改方便掺入参数

#这只事构建了网络还需要写训练代码


'''
Pre-training Trajectory Model
'''
Trajectory_Duplicated = Layer2trajectory(n_inputpart1=操作轨迹的数目还需看论文需要对数据进行预处理,n_hiddenpart1=100,n_inputpart2=相似操作轨迹的数目还需要看论文对数据进行处理,n_hiddenpart2=100,n_inputpart2=最反常操作轨迹的数目还需要看论文对数据进行处理,n_hiddenpart3=100,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1=100,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
#这只事构建了网络还需要写训练代码


'''
Deep Multimodal Embedding
'''
#构建神经网络的架构
RoboaristNet = Layer3(n_inputpart1=云点的数目还需看论文需要对数据进行预处理,n_hiddenpart1=250,n_inputpart2=语音的数目还需要看论文对数据进行处理,n_hiddenpart2=150,n_inputpart3=最反常语音的数目还需要看论文对数据进行处理,n_hiddenpart3=150,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01),scale=0.1,n_hidden1=125,n_hidden1_tr =100,n_hidden2=25,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama,n_input= 操作轨迹的数目还需看论文需要对数据进行预处理,n_hidden = 100)

#初始化神经网络会话
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print('begin to run session... ...')
#记录训练日志
writer = tf.summary.FileWriter(logdir='logs', graph=RoboaristNet.sess.graph)
writer.close()
#预处理训练数据
def Preprocessor(相应参数):
    预处理函数
    return 相应的函数结果
def Get_random_minibatch(相应参数):
    抓取一个minibatch的数据
    return 相应输出结果
def Number_train(相应参数):
    return 相应输出结果     # 训练样本总数
def Randomoneinput(相应参数)：
    return 相应输出为批次中的随机一个（无放回）
Training_epochs = 根据论文填写相应的循环训练次数                          # 训练轮数，1轮等于n_samples/batch_size
Mini_batch_size = 根据论文填写                              # batch容量
display_step = 1 #现实训练结果的参数

#训练网络
for epoch in range(Training_epochs):
    all_cost = 0
    all_avg_cost = 0                              # 一个minibatch总损失
    total_batch = int(Number_train函数返回值/Mini_batch_size)   # 每一轮中step总数

    for i in range(total_batch):
        batch_xs = Get_random_minibatch(相应参数)#获得批次训练的数据
        for _ in range(Mini_batch_size):
           Randomoneinput = Randomoneinput(相应参数batch_xs)
           cost = RoboaristNet.Caculateloss(Randomoneinput)#计算出
           all_cost += cost 
        avg_cost = all_cost / Mini_batch_size
        RoboaristNet.Train(avg_cost)#The average cost of each minibatch is back-propagated through all the layers of the deep neural network using the AdaDelta [29] algorithm.   
        #all_avg_cost += avg_cost
    avg_cost_final = avg_cost
 
    if epoch % display_step == 0:
        print('epoch : %04d, cost = %.9f' % (epoch+1, avg_cost_final)) 

print('Total coat:', str(RoboaristNet.Caculateloss(X_test))) #计算出测试集的损失值      
#这只事构建了网络还需要写训练代码


'''
Roboaristanetpointcloudlanguage = Layer3_pointcloudlanguage(n_inputpart1=云点的数目还需看论文需要对数据进行预处理,n_hiddenpart1=250,n_inputpart2=语音的数目还需要看论文对数据进行处理,n_hiddenpart2=150,n_inputpart3=最反常语音的数目还需要看论文对数据进行处理,n_hiddenpart3=150,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1=125,n_hidden2=25,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)

Roboaristanettrajectory = Layer3_trajectory(n_input= 操作轨迹的数目还需看论文需要对数据进行预处理,n_hidden = 100,activate_function=tf.nn.relu,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1,n_hidden1 = 100,n_hidden2 = 25,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
'''


'''
sume = 0
with tf.name_scope('Pretraining'):
     for i in range(minibatch):#这是输入的minibatch的对数（pointcloud和language）
        
        sume += self.losslayer2
     Pretraining = tf.train.AdadeltaOptimizer(0.1).minimize(losslayer2)
with tf.name_scope('Finetune'):
     Finetune = tf.train.AdadeltaOptimizer(0.01).minimize(self.losslayer3)
'''






'''

#main function to use the class 
Point_cloud_encoder = SDA(n_input=云点的数目还需看论文，n_hidden=250,activate_function=tf.nn.relu,optimzer=tf.train.AdadeltaOptimizer(learning_rate=0.01),scale=0.01)#initialize the network with the parameters

writer = tf.summary.FileWriter(logdir='logs',graph=Point_cloud_encoder.sess.graph)
writer.close()#display the graph of the network 


def load_input_data(数据的地址，需要加载成的数据格式)
    return data
number_data = 具体根据论文填写#this is the number of all input data 
training_epochs = 具体根据论文填#this is the number of iterate
batch_size = 具体根据论文填#this is the number of the batch_size
distply_step = 1
#begin real training
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(number_data/batch_size)
    for i in range(total_batch)
        batch_inputs = get_random_block_from_inputs_data(data,batch_size)
        cost = Point_cloud_encoder.partial_fit(batch_inputs)
        avg_cost += cost / batch_size
    avg_cost /= total_batxh
    
    if epoch % display_step == 0:
        print('epoch : %04d, cost = %.9f' % (epoch+1, avg_cost))

print('Total coat:', str(Point_cloud_encoder.calc_cost(X_test)))


'''

'''     
      #two mode of training
    def partial_fit(self,inputs):#define a function for training and caculate the loss and optimizer
        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict={self.input:inputs, self.scale:scale}) 
        return  loss
    
    def calc_cost(self,inputs):#define a function only caculate the loss
        only_loss = self.sess.run(self.loss,feed_dict={self.input:inputs,self.scale:scale})
    def get_random_size_from_data(data，batch_size):#data is number all the input
        start_index = np.random.randint(0, len(data)-batch_size)
        return data[start_index:(start_index+batch_size)]
'''






