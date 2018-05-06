#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
L h 2,pl (p i , l i , τ i ) = |∆(τ i , τ )+sim(Φ P (p i ), Φ L (l ))
−sim(Φ P (p i ), Φ L (l i ))| +

'''


#导入相应的文件包
from argmax_l import Argmax_l
from argmax_tr import Argmax_tr
from DTW_WT_value import D_matricx_value

class Loss_h2_pl(object):
     def __init__(self,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
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
     def _most_violating_language(self):
         language_a_init = Argmax_l(self.pointcloud_i,self.most_fit_trajectory_star这个是预先知道的配套于云点与语音,self.trajectory_Set_I这是所有的可能轨迹集合,self.language_Set 这是语言集合,self.tS这是一个阀值,self.tD这是一个阀值,self.alpha_t,self.alpha_r，self.beta，self.gama)
         language_a = trajectory_a_init.most_violating_trajectory()#这将计算出t中最反常的
         return language_a

     def _most_violating_trjectory(self):
         trajectory_b_init = Argmax_tr(self.pointcloud_i,self.language_i,self.most_fit_trajectory_star这个是预先知道的配套于云点与语音,self.trajectory_Set_I这是所有的可能轨迹集合,self.tS,self.alpha_t，self.alpha_r，self.beta，self.gama,self.alpha)
         trajectory_b = trajectory_b_init.most_violating_language() #需要将这个函数写成类将其计算出l中最反常的
         return trajectory_b#hidden_layer1_argmax_language = Argmax_l #需要将这个函数写成类将其计算出l i中最反常的
     def _trajectory_random_in_simillar(self,numbera):
         simillar_trjectory_set = 选取相关轨迹的函数(self.trajectory_Set_I,self.tS这是一个阀值)
         value_trajectory_random_in_simillar = simillar_trjectory_set[numbera]
         return value_trajectory_random_in_simillar

     def _value_D_MATRICX(self,number):
         value_D_MATRICX_init =  D_matricx_value(self._trajectory_random_in_simillar(number),self._most_violating_trjectory(),self.alpha_t,self.alpha_r,self.beta，self.gama)#.normalize_D_matricx()
         value_D_MATRICX_number = value_D_MATRICX_init.normalize_D_matricx()#注意L h 2,pl (p i , l i , τ i ) = |∆(τ i , τ )+sim(Φ P (p i ), Φ L (l ))−sim(Φ P (p i ), Φ L (l i ))| +中的τ i是属于相关集合中的但是是否就确认就是这个与云点与语言的值，还需询问作者
         return value_D_MATRICX_numbe

     def _hidden_layer1_pointcloud(self,second_hidden_pointcloud):
         return second_hidden_pointcloud#计算出第一层中point_cloud_i的值(self.pointcloud_i)

     def _hidden_layer1_language(self,second_hidden_language):
         return second_hidden_language#计算出第一层中language_i的值(self.language_i)
 
     def _hidden_layer1_language_violating(self,second_hidden_most_violating_language):
         return second_hidden_most_violating_language#计算出第一层中language_violating的值(self._most_violating_language())
         
#hidden_layer1_argmax_language = Argmax_l #需要将这个函数写成类将其计算出l i中最反常的
     def _similarity_pi_argmax_l(self,point_layer2,language_violating_layer2):#输入的应当是第二层的分量 #Φ P (p) : P → R N 2,p  ,Φ L (l) : L → R N 2,pl并且是最反常的语言
         value_similarity_pi_argmax_l = 向量相乘(self._hidden_layer1_pointcloud(point_layer2),self._hidden_layer1_language_violating(language_violating_layer2))
         return value_similarity_pi_argmax_l

     def _similarity_pi_li(self,point_layer2,language_layer2):
         value_similarity_pi_li = 向量相乘(self._hidden_layer1_pointcloud(point_layer2),self._hidden_layer1_language(language_layer2))
         return value_similarity_pi_li

     def loss_function_h2(self,i,j,k,l):
         sum_loss = self._value_D_MATRICX(i) + self._similarity_pi_argmax_l(j,k) + self._similarity_pi_li(j,l)
         value_loss_function = 求取平方后然后开根号取正(sum_loss)
         return value_loss_function
     



