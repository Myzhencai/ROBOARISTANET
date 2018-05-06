#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
τ i = arg max (sim(Φ P,L (p i , l i ), Φ T (τ )) + α∆(τ i , τ ))

this file is for the argmax_tr



'''
#导入你所需要的函数包需要改成类
from DTW_WT_value import D_matricx_value


class Argmax_tr(object):
     def __init__(self,pointcloud_i,language_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,tS,alpha_t，alpha_r，beta，gama,alpha):
         self.pointcloud_i = pointcloud_i
         self.language_i = language_i
         self.most_fit_trajectory_star = most_fit_trajectory_star
         self.trajectory_Set_I = trajectory_Set_I
         self.tS = tS
         self.tD = tD
         self.alpha_t = alpha_t
         self.alpha_r = alpha_r
         self.beta = beta
         self.gama = gama
         self.alpha = alpha

#这两个是第二层神经网络的分值，二者相加是第二层的结果
     def _seperate_set_simillar(self):
         trajectory_Set_I_similar = 计算出相同轨迹集合的函数(self.most_fit_trajectory_star,self.tS这是一个阀值,self.trajectory_Set_I = trajectory_Set_I)#T i,S = {τ ∈ T |∆(τ i ∗ , τ ) < t S }
         return trajectory_Set_I_similar

     def _seperate_set_difference(self):
         trajectory_Set_I_difference = 计算出不相同轨迹集合的函数(self.most_fit_trajectory_star,self.tD这是一个阀值,self.trajectory_Set_I = trajectory_Set_I)# T i,D = {τ ∈ T |∆(τ i ∗ , τ ) > t D }
         return trajectory_Set_I_difference

     def _random_trajectory_Set_I_difference(self,randoma):
         trajectory_Set_I_difference = self._seperate_set_difference()
         return trajectory_Set_I_difference[randoma]#返回_random_trajectory_Set_I_difference的第randoma个

     def _random_trajectory_Set_I_simillar(self,randomb):
         trajectory_Set_I_simillar = self._seperate_set_simillar()
         return trajectory_Set_I_difference[randomb]

     def _point_cloud_pl_2_merge_map_to_h3(self):#hidden_layer1_pointcloud_2,hidden_layer1_language_2):
         return 神经网络中云点与语音合成后的输出最后映射到h3层的embedding值(self.pointcloud_i,self.language_i)#Φ P,L (p i , l i )


     def _trajectory_pl_2_merge_map_to_h3_difference(self,random_number0):
         return 神经网络中云点与语音合成后的输出最后映射到h3层的embedding值(self._random_trajectory_Set_I_difference(random_number0))#Φ T (τ)中随机的一个的第三城的embedding值
'''
def _trajectory_pl_2_merge_map_to_h3_simillar(self,random_numberb):
    return 神经网络中云点与语音合成后的输出最后映射到h3层的embedding值(self._random_trajectory_Set_I_simillar(random_numberb))
'''


     def _similarity_point_language_trajectory_3(self,random_number1):
         value_similarity_point_language_trajectory_3 = 向量相乘函数(self._point_cloud_pl_2_merge_map_to_h3(),self._trajectory_pl_2_merge_map_to_h3(random_number1))
         return value_similarity_point_language_trajectory_3#sim(Φ P,L (p i , l i ), Φ T (τ ),Φ T (τ)中随机的一个

 

     def _Dmatricx_value(self,random_numbera,random_numberb): #注意这里倒入的两个轨迹是有要求的一个是与云点语音关联的一个是不关联的并且不相关的是没有编号的是要枚举的
         D_matricx_value_init = D_matricx_value(self._random_trajectory_Set_I_difference(random_numbera),self._random_trajectory_Set_I_simillar(random_numberb),self.alpha_t，self.alpha_r，self.beta，self.gama)
         value_Dmatricx_value = D_matricx_value_init.normalize_D_matricx()
         last_value_Dmatricx_value = 将一个数值乘如一个矩阵函数(self.alpha , value_Dmatricx_value)
         return last_value_Dmatricx_value

     def _optimizer_need_part(self):
    #number_set1 = 计算集合元素个数的函数（一个集合L）
         number_set2 = 计算集合元素个数的函数（self._seperate_set_difference()）
         number_set3 = 计算集合元素个数的函数（self._seperate_set_simillar()）#由此可见∆(τ, τ i ∗ )应该是个数值而不是一个矩阵，后期还要更改
         d1 = 1#应该是个可以在下面应用的变量
         d2 = 1
    #d3 = 1
    #d4 = 1
         e1 = 1
    #e3 = 1
    #f1 = 1
        for b in range(number_set2):

           for c in range(number_set3):

               larger_line = self._similarity_point_language_trajectory_3(b) + self._Dmatricx_value(d1,b)#Dmatricx_value(trajectory_Set_I_similar[d1],trajectory_Set_I_difference[b])

               smaller_line = self._similarity_point_language_trajectory_3(b) + self._Dmatricx_value(c+1,b)#Dmatricx_value(trajectory_Set_I_similar[c+1],trajectory_Set_I_difference[b]) #语音集合中的第一层中的第a个
               if (larger_line >= smaller_line ):#这里通过改变d的数值来保存最大值
                  d1 += 0
                  #d1 = d
               else
                  d1 += 1
                  #d1 = d
               c += 1
               return d1
            #这是算一条线的最大值相当于larger_line
            larger_face = self._similarity_point_language_trajectory_3(e1) + self._Dmatricx_value(d1,e1)#Dmatricx_value(trajectory_Set_I_similar[d1],trajectory_Set_I_difference[e1])

            #这就相当于 smaller_line          
            for c in range(number_set3):

                larger_line = self._similarity_point_language_trajectory_3(b+1) + self._Dmatricx_value(d2,b+1)#Dmatricx_value(trajectory_Set_I_similar[d2],trajectory_Set_I_difference[b+1])

                smaller_line = self._similarity_point_language_trajectory_3(b+1) + self._Dmatricx_value(c+1,b+1)#Dmatricx_value(trajectory_Set_I_similar[c+1],trajectory_Set_I_difference[b+1])
                if (larger_line >= smaller_line ):
                   d2 += 0
                  #d2 = d
                else 
                   d2 += 1
                  #d2 = d
                c += 1
                return d2
            smaller_face = self._similarity_point_language_trajectory_3(b+1) + self._Dmatricx_value(d2,b+1)#Dmatricx_value(trajectory_Set_I_similar[d2],trajectory_Set_I_difference[b+1])
 
         if (larger_face >= smaller_face ):
            e1 += 0
            d1 += 0
                  #d2 = d
         else 
            e1 += 1
            d1 = d2
         b += 1
         return e1 

     def most_violating_language(self):
         value_most_violating_language_index = self._optimizer_need_part()
         value_most_violating_language = self._random_trajectory_Set_I_difference(value_most_violating_language_index)#获得不同轨迹的最反常的那个 
         return value_most_violating_language


    



