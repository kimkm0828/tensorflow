import tensorflow as tf


a = tf.constant(120,name='a')
b = tf.constant(130,name='b')
c = tf.constant(140,name='c')

# 변수 정의
v = tf.Variable(0, name='v')

# 데이터 플로우 그래프 정의하기
calc_op = a + b + c
assgin_op = tf.assign(v, calc_op)

# 세션 실행하기
sess = tf.Session()
sess.run(assgin_op)



print(sess.run(v))


'''
    왜 v에 넣냐
    
    x = a + b
    이거 생각하면 끝
'''
























