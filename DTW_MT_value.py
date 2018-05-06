#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
We use the DTW-MT distance function (de-scribed later in Sec. V-C) for our loss function ∆(τ, τ  ̄ ), but
it could be replaced by any function that computes the lossof predicting τ  ̄ when τ is the correct demonstration.
this class is for value_D_matricx and then normalize it to get the ∆(τ j , τ k )
'''

#define a function to caculate the column D(i, 1)
from DTW_WT import DTW_WT

class D_matricx_value(object):
     def __init__(self,trajectory_a,trajectory_b,alpha_t,alpha_r,beta，gama)
        self.trajectory_a = trajectory_a
        self.trajectory_b = trajectory_b
        self.number_trajectory_a = 计算trajectory_a的轨迹点个数
        self.number_trajectory_b = 计算trajectory_a的轨迹点个数
        self.alpha_r = alpha_r
        self.alpha_t = alpha_t 
        self.beta = beta
        self.gama = gama
        self.D_MATRICX = 给提供一个函数用于生成 self.number_trajectory_a * self.number_trajectory_b 的矩阵
        #self.i = i
        #self.j = j
        #eve_value_D_MATRICX = self._rarel_d(self.i,self.j)
        
        #self.eve_value_D_MATRICX = eve_value_D_MATRICX 
        
     def _column_d(self,i,j)#i,j,trajectory_a,trajectory_b):
        if (j == 1 and i <= self.number_trajectory_a):#need to caculate the number_trajectory_a with a function假设已经计算出来了
        DTW_WT_init = DTW_WT(self.trajectory_a,self.trajectory_b,1,1，self.alpha_t，self.alpha_r，self.beta，self.gama)
        value_column_d = DTW_WT_init
        for a in range(i):
            a +=1
            value_column_d +=DTW_WT.c_function(self.trajectory_a,self.trajectory_b,a,1，self.alpha_t，self.alpha_r，self.beta，self.gama)#D(i, 1)
        return value_column_d
        else 
        print('error')

#define a function to caculate the column D(1, j)                      
     def _row_d(self,i,j):
         if (i == 1 and j <= self.number_trajectory_b):#need to caculate the number_trajectory_a with a function假设已经计算出来了
         DTW_WT_init = DTW_WT(self.trajectory_a,self.trajectory_b,1,1，self.alpha_t，self.alpha_r，self.beta，self.gama)
         value_row_d = DTW_WT_init
         for a in range(j):
            a +=1
            DTW_WT_add = DTW_WT(self.trajectory_a,self.trajectory_b,a,1，self.alpha_t，self.alpha_r，self.beta，self.gama)
            value_column_d +=DTW_WT_add.c_function()#D(1, j)
         return value_row_d
         else 
         print('error')
                    
#define a function to caculate the column D(i, j)
    def _rarel_d(self,i,j)#trajectory_a,trajectory_b)
        DTW_WT_init = DTW_WT(self.trajectory_a,self.trajectory_b,i,j，self.alpha_t，self.alpha_r，self.beta，self.gama)
        part0_value_rarel_d = DTW_WT_init.c_function()#c(τ A , τ B )
        if (j == 1 and i <= self.number_trajectory_a):
        value_rarel_d = self._column_d(i,j)
        return value_rarel_d

        elif(i == 1 and j <= self.number_trajectory_b):
        value_rarel_d = self._row_d(i,j)
        return value_rarel_d

        elif(i == 1 and j == 1):#we are not sure that this is the exact result of the base so we need to think more of the 
        value_rarel_d = part0_value_rarel_d
        return value_rarel_d

        else
        value_rarel_d = part0_value_rarel_d + 取最小值函数(rarel_d(i-1,j-1),rarel_d(i-1,j),rarel_d(i,j-1))#for those whose value turn to 1 we need to chage the way to caculate it and we need garantee that the value at least be one
        return value_rarel_d

   def _fill_in_matricx(self):
       for a in range(self.number_trajectory_a):
          for b in range(self.number_trajectory_b):
              self.D_MATRICX[a,b] = self._rarel_d(a,b)
              b +=1
           a +=1
           value_D_matricx =  self.D_MATRICX
       return  value_D_matricx
   def normalize_D_matricx(self):
       value = self._fill_in_matricx()
       value_normalize_D_matricx = 计算矩阵归一化的函数(value)



'''  
if (i != 1 and j !=1):
      DTW_WT_init = DTW_WT(trajectory_a,trajectory_b,i,j，alpha_t，alpha_r，beta，gama)
      part0_value_rarel_d = DTW_WT_init.c_function()#c(τ A , τ B )
      
      DTW_WT_final = part0_value_rarel_d + 取最小值的函数()
''' 
         

 














         
