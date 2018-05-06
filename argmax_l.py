#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
First, we pre-train the layers leading up to these layers
using sparse de-noising autoencoders [30], [31]. Then, our
process for pre-training h 2,pl is similar to our approach to
fine-tuning a semantically meaningful embedding space for
h 3 presented above, except now we find the most violating
language l while still relying on a loss over the associated
optimal trajectory:
l = argmax (sim(Φ P (p i ), Φ L (l)) + α∆(τ i , τ ))

and will return the largest value of it and the location of it 

this file is for the argmax_l



'''
#导入你所需要的函数包需要改成类
from DTW_WT_value import D_matricx_value


class Argmax_l(object):
     def __init__(self,pointcloud_i,most_fit_trajectory_star这个是预先知道的配套于云点与语音,trajectory_Set_I这是所有的可能轨迹集合,language_Set这是语言集合,tS这是一个阀值,tD这是一个阀值,alpha,alpha_t，alpha_r，beta，gama)
        self.pointcloud_i = pointcloud_i
        #self.language_i = language_i
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

#这两个是第二层神经网络的分值，二者相加是第二层的结果
     def _seperate_set_similar(self):
         trajectory_Set_I_similar = 计算出相同轨迹集合的函数(self.most_fit_trajectory_star,self.tS这是一个阀值,self.trajectory_Set_I )#T i,S = {τ ∈ T |∆(τ i ∗ , τ ) < t S }
         return trajectory_Set_I_similar
     def _seperate_set_difference(self):
         trajectory_Set_I_difference = 计算出不相同轨迹集合的函数(self.most_fit_trajectory_star,self.tD这是一个阀值,self.trajectory_Set_I)# T i,D = {τ ∈ T |∆(τ i ∗ , τ ) > t D }
         return trajectory_Set_I_difference

     def _point_cloud_pl_2_pointcloud(self):#只能是i组中的云点
         return 神经网络第N2_pl层的输出分量(self.pointcloud_i)
     def _point_cloud_pl_2_language(self,random):#是所有语音集合中的随机一个
         return 神经网络第N2_pl层的输出分量(self.language_Set[random])#random 是相应的值

     def _similarity_point_language_pl_2(self,random_number):#point_cloud_pl_2(hidden_layer1_pointcloud_i),point_cloud_pl_2(hidden_layer1_laguage_random)):
         value_similarity_point_language_pl_2 = 向量相乘函数(self._point_cloud_pl_2_pointcloud(),self._point_cloud_pl_2_language(random_number))#sim(Φ P (p i ), Φ L (l))
         return value_similarity_point_language_pl_2 

     def _trajectory_i_in_S(self,i):
         return _seperate_set_similar[i]#返回集合中的第i个
   
     def _trajectory_i_in_D(self,j):
         return _seperate_set_difference[j]#返回集合中的第j个
         

     def _Dmatricx_value(self,i,j): #注意这里倒入的两个轨迹是有要求的一个是与云点语音关联的一个是不关联的并且不相关的是没有编号的是要枚举的v  #注意两篇论文中的t一个时随意的一个是最适合的
         D_matricx_value_init = D_matricx_value(self._trajectory_i_in_S(i),self._trajectory_i_in_D(j),self.alpha_t，self.alpha_r，self.beta，self.gama)
         value_Dmatricx_value = D_matricx_value_init.normalize_D_matricx()
         last_value_Dmatricx_value = 将一个数值乘上一个矩阵函数(self.alpha , value_Dmatricx_value)#应该是计算的不是矩阵而是一个数值后期需要询问作者
         return last_value_Dmatricx_value

    def _optimizer_need_part(self):
        number_set1 = 计算集合元素个数的函数（self.language_Set）
        number_set2 = 计算集合元素个数的函数（self._seperate_set_difference()）
        number_set3 = 计算集合元素个数的函数（self._seperate_set_similar()）#由此可见∆(τ, τ i ∗ )应该是个数值而不是一个矩阵，后期还要更改
        d1 = 1#应该是个可以在下面应用的变量
        d2 = 1
        d3 = 1
        d4 = 1
        e1 = 1
        e3 = 1
        f1 = 1
        for a in range(number_set1):

            for b in range(number_set2):

               for c in range(number_set3):

                   larger_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d1,b)

                   smaller_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(c+1,b) #Dmatricx_value(trajectory_Set_I_similar[c+1],trajectory_Set_I_difference[b]) #语音集合中的第一层中的第a个
                   if (larger_line >= smaller_line ):#这里通过改变d的数值来保存最大值
                      d1 += 0
                  #d1 = d
                   else 
                      d1 = c+1
                  #d1 = d
                   c += 1
                   return d1
            #这是算一条线的最大值相当于larger_line
               larger_face = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d1,e1) #Dmatricx_value(trajectory_Set_I_similar[d1],trajectory_Set_I_difference[e1])

            #这就相当于 smaller_line          
               for c in range(number_set3):

                  larger_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d2,b+1)#Dmatricx_value(trajectory_Set_I_similar[d2],trajectory_Set_I_difference[b+1])

                  smaller_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(c+1,b+1)#Dmatricx_value(trajectory_Set_I_similar[c+1],trajectory_Set_I_difference[b+1]) 
                  if (larger_line >= smaller_line ):
                     d2 += 0
                  #d2 = d
                  else 
                     d2 = c+1 #将c+1的值赋给d2
                  #d2 = d
                  c += 1
                  return d2
               smaller_face = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d2,b+1)#Dmatricx_value(trajectory_Set_I_similar[d2],trajectory_Set_I_difference[b+1])

               if (larger_face >= smaller_face ):#这里通过改变d的数值来保存最大值
                   e1 += 0
                   d1 += 0
                  #d1 = d
               else 
                  e1 += b+1
                  d1 = d2#将都的值赋给d1
                  #d1 = d
               b += 1
               return e1 ， d1

            larger_cube = self._similarity_point_language_pl_2(f1) + self._Dmatricx_value(d1,e1)#Dmatricx_value(trajectory_Set_I_similar[d1],trajectory_Set_I_difference[e1])

            for b in range(number_set2):

                for c in range(number_set3):

                   larger_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d3,b)#Dmatricx_value(trajectory_Set_I_similar[d3],trajectory_Set_I_difference[b])

                   smaller_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(c+1,b)#Dmatricx_value(trajectory_Set_I_similar[c+1],trajectory_Set_I_difference[b]) #语音集合中的第一层中的第a个
                   if (larger_line >= smaller_line ):#这里通过改变d的数值来保存最大值
                      d3 += 0
                  #d3 = d
                   else 
                      d3 = c+1
                  #d3 = d
                   c += 1
                   return d3
            #这是算一条线的最大值相当于larger_line
                larger_face = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d3,e3)#Dmatricx_value(trajectory_Set_I_similar[d3],trajectory_Set_I_difference[e3])

            #这就相当于 smaller_line          
                for c in range(number_set3):

                   larger_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d4,b+1)#Dmatricx_value(trajectory_Set_I_similar[d4],trajectory_Set_I_difference[b+1])

                   smaller_line = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(c+1,b+1)#Dmatricx_value(trajectory_Set_I_similar[c+1],trajectory_Set_I_difference[b+1]) 
                   if (larger_line >= smaller_line ):
                      d4 += 0
                  #d4 = d
                   else 
                      d4 = c+1
                  #d4 = d
                   c += 1
                   return d4
                smaller_face = self._similarity_point_language_pl_2(a) + self._Dmatricx_value(d4,b+1)#Dmatricx_value(trajectory_Set_I_similar[d4],trajectory_Set_I_difference[b+1])

                if (larger_face >= smaller_face ):#这里通过改变d的数值来保存最大值
                     e3 += 0
                     d3 += 0
                  #d3 = d
                else 
                    e3 += b+1
                    d3 = d4#将都的值赋给d3
                  #d3 = d
                b += 1
            return e3 ， d3

            smaller_cube = self._similarity_point_language_pl_2(a+1) + self._Dmatricx_value(d3,e3)# + Dmatricx_value(trajectory_Set_I_similar[d3],trajectory_Set_I_difference[e3])    
          


       if (larger_cube >= smaller_cube ):#这里通过改变d的数值来保存最大值
          e1 += 0
          d1 += 0
          f1 += 0
                  #d3 = d
       else 
          e1 = e3
          d1 = d3#将都的值赋给d3
          f1 = a+1 
       a += 1
       return f1       
       
   def most_violating_trajectory(self):
       value_most_violating_language_index = self._optimizer_need_part()
       value_most_violating_language = language_Set[f1]
       return value_most_violating_language


#most_violating_language =  一个集合L[f1]



    



