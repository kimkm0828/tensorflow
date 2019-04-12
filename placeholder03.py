import tensorflow as tf
import numpy as np

'''
    
'''
student = [
    ['김경민',80,90,100],
    ['강감찬',80,70,50],
    ['유관순',80,50,90],
    ['문제인',80,60,70],
    ['이순신',100,90,100]
    ]
ph_score = tf.placeholder(tf.int32,[None])
cnt = tf.constant(3)
# avg = (ph_score[0] + ph_score[1] + ph_score[2]) /cnt
avg = tf.reduce_mean(ph_score)


sess = tf.Session()

for i in range(len(student)):
    r1 = sess.run(avg,feed_dict={ ph_score:student[i][1:4] })
    print(student[i][0],r1)









# a = tf.placeholder(tf.int32,[None])
# b = tf.constant(3)
# avg_op = a / b
#
# sess = tf.Session()
# for i in range(len(student)):
#
#     r1 = sess.run(avg_op,feed_dict={ a:student[i][1:4] })
#     print(r1)




