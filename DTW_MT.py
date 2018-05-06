#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
We use the DTW-MT distance function (de-scribed later in Sec. V-C) for our loss function ∆(τ, τ  ̄ ), but
it could be replaced by any function that computes the lossof predicting τ  ̄ when τ is the correct demonstration.
this code is for cacuclate the c(τ A , τ B ; α T , α R , β , γ)
'''
#import the dic that you need 
import tensorflow as tf


#define the class for DTW_MT distance function this function is for c(τ A , τ B ; α T , α R , β , γ)
class DTW_WT(object):
     def __init__(self,trajectory_a,trajectory_b,number_i,number_j，alpha_t，alpha_r，beta，gama):#the trajecrory_a and trajectory_b for use should make a function let them in
        self.trajectory_a = trajectory_a
        self.trajectory_b = trajectory_b
        self.i = number_i
        self.j = number_j
        self.alpha_t = alpha_t#随机给数据#may need the network train for it 
        self.alpha_r = alpha_r#随机给数据
        #self.n_trajectory_a = 用tensorflow计算一个矩阵的行数
        #self.n_trajectory_b = 用tensorflow计算一个矩阵的行数
        #self.D_matrix = 用tensorflow定义一个n_trajectory_a * n_trajectory_b 的空矩阵#since this are all matricx I think we can use other dic for use
        self.beta = beta#随机给数据
        self.gama = gama#随机给数据
        
'''
        value_Wa_trajectory = self._W_trajectory(trajectory[i],gama)
        self.value_Wa_trajectory = value_Wa_trajectory

        value_Wb_trajectory = self._Wb_trajectory(trajectory[j],gama)
        self.value_Wb_trajectory = value_Wb_trajectory

        value_bool_griper_station = self._bool_griper_station(trajectory_a[i],trajectory_b[j])
        self.value_bool_griper_station = value_bool_griper_station

        value_angle_difference = self._angle_difference(trajectory_a[i],trajectory_b[j])
        self.value_angle_difference = value_angle_difference

        value_trajetory_difference = self._trajetory_difference(trajectory_a[i],trajectory_b[j])
        self.value_trajetory_difference = value_trajetory_difference

        value_c_function = self._c_function(value_Wa_trajectory,value_Wb_trajectory,value_trajetory_difference,
                                           value_angle_difference,value_bool_griper_station,alpha_t,alpha_r,beta)

        self.value_cij_function = value_c_function
'''

   #define the function for w(τ (i) ; γ) = exp(−γ · ||τ (i) || 2 )
    def _Wa_trajectory(self,i):#trajectory,i_number,gama):
      '''
       i_number = self.i
       gama = self.gama
       trajectory = self.trajectory_a
      '''
       trajectory_2_norm =  ||self.trajectory_a[i]|| 2计算向量2-范数值#use some special dic for this 
       index_number = 取反函数（self.gama * trajectory_2_norm）#you can change the self
       value_W_trajectory = 指数函数(index_number)
       return value_Wa_trajectory

   #define the function for w(τ (i) ; γ) = exp(−γ · ||τ (i) || 2 )
    def _Wb_trajectory(self,j):#trajectory,j_number,gama):
       '''
       i_number = self.j
       gama = self.gama
       trajectory = self.trajectory_b
       '''
       trajectory_2_norm =  ||self.trajectory_b[j]|| 2计算向量2-范数值#use some special dic for this 
       index_number = 取反函数（self.gama * trajectory_2_norm）#you can change the self
       value_W_trajectory = 指数函数(index_number)
       return value_Wb_trajectory

   #define the function for d G (τ A , τ B ) = 1 (g A = g B )
    def _bool_griper_station(self,i,j):#,trajectory_a[i],trajectory_b[j]):#this should be a bool function maybe we need to modify it to something cooler
       if self.trajectory_a[i] = self.trajectory_b[j]:
          return 1
       else:
          return 0

   #define the function for d R (τ A , τ B ) = angle difference between τ A and τ B
    def _angle_difference(self,i,j):#,trajectory_a[self.i],trajectory_b[self.j]):# for the reason we need the trajectory_a[i],trajectory_b[j] so it should be in the self.value in class and need a function for extract the value of it !!!
       angle_trajectory_a = 提取角度函数(self.trajectory_a[i])
       angle_trajectory_b = 提取角度函数(self.trajectory_b[j])
       value_angle_difference = 相减函数(angle_trajectory_a,angle_trajectory_b)
       return value_angle_difference

   #define the function for d T (τ A , τ B ) = ||(t x ,t y ,t z ) A − (t x ,t y ,t z ) B || 2
    def _trajetory_difference(self,i,j):#,trajectory_a[self.i],trajectory_b[self.j]):
       trajectory_position_a = 提取坐标值的函数(self.trajectory_a[i])
       trajectory_position_b = 提取坐标值的函数(self.trajectory_b[j])
       sub_trajectory = trajectory_position_a - trajectory_position_b
       value_trajetory_difference = 计算2-范数的函数(sub_trajectory)
       return value_trajetory_difference

   #define the function for the c_function
    def c_function(self)#i,j):#,w_a,w_b,dt,dr,dg,self.alpha_t,self.alpha_r,self.beta):
       i = self.i
       j = self.j
       dt_alpha = self._trajetory_difference(i,j) / self.alpha_t# they are all matricx 
       dr_alpha = self._angle_difference(i,j) / self.alpha_r
       add_dr_dt = dt_alpha + dr_alpha#matricx add !!!
       one_dg = 1 + self._bool_griper_station(i,j) * self.beta
       w_a = self._Wa_trajectory(i)
       w_b = self._Wb_trajectory(j)
       value_c_function = 矩阵相乘(w_a,w_b,add_dr_dt)*one_dg
       return value_c_function
          
       

