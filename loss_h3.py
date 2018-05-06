#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
L h 3 (p i , l i , τ i ) = |∆(τ i , τ i )+sim(Φ P,L (p i , l i ), Φ T (τ i ))
−sim(Φ P,L (p i , l i ), Φ T (τ i ))| +
'''
from argmax_l import Argmax_l
from argmax_tr import Argmax_t
from DTW_WT_value import D_matricx_value
#还需要修改为类

trajectory_b = Argmax_t #需要将这个函数写成类将其计算出τ i中最反常的
value_D_MATRICX =  D_matricx_value(trajectory_i,trajectory_b,alpha_t,alpha_r,beta，gama).normalize_D_matricx()
class Loss_h3(object):
def similarity_pi_li_argmax_t(hidden_layer2_pointcloud_i_language_i , hidden_layer2_argmax_trajectory):#输入的应当是第3层云点与语音的embedding和最反常轨迹的第三层embeddingsim(Φ P,L (p i , l i ), Φ T (τ i )
    value_similarity_pi_li_argmax_t = 向量相乘(hidden_layer2_pointcloud_i_language_i,hidden_layer2_argmax_trajectory)
    return value_similarity_pi_li_argmax_t

def similarity_pi_li_ti(hidden_layer2_pointcloud_i_language_i , hidden_layer2_trajectory_i):#输入的应当是第3层云点与语音的embedding和轨迹的第三层embeddingsim(Φ P,L (p i , l i ), Φ T (τ i )
    value_similarity_pi_li_ti = 向量相乘(hidden_layer2_pointcloud_i_language_i,hidden_layer2_trajectory_i)
    return similarity_pi_li_ti

def loss_function_h3(point_cloud_i ,language_i ,trajectory_i ):
    sum_loss = value_D_MATRICX + similarity_pi_li_argmax_t - similarity_pi_li_ti
    value_loss_function = 求取平方后然后开根号取正(sum_loss)
    return value_loss_function
