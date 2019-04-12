#!/usr/bin/env python
# coding: utf-8

# # 주피터 노트북에서 tensorflow 사용하기
# 주피터 노트북에서 사용해보자

# In[1]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[5]:


import tensorflow as tf


# In[6]:


import pandas as pd


# In[8]:


# 연산 정의
a = tf.constant(50)
b = tf.constant(100)
calc1 = a + b
# 바로 출력안됨

# 변수만들기
v = tf.Variable(0)



# 연산결과를 텐서변수 v에 대입
let_op = tf.assign(v, calc1)


# In[9]:



sess = tf.Session()


sess.run(tf.global_variables_initializer())


sess.run(let_op)


print(sess.run(v))






