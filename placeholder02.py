import tensorflow as tf

# 크기에 상관 없는 정수형 1차원 배열의 placeholder를 만들고
# 곱하기 2한 결과를 텐서이용하여 출력

arr = tf.placeholder(tf.int32,shape=None,name='arr')

b = tf.constant(2)
x_op = arr * b


sess = tf.Session()

r1 = sess.run(x_op,feed_dict={arr:[3,4,5,6,7]})
print(r1)








