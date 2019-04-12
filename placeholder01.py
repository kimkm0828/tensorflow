import tensorflow as tf

'''
    마치 데이터 베이스의 PrepareStatment처럼
    질의문의 ?를 생각하면 됨
    사용자가 입력한 값을 질의문에 대응시키려고 ?를 대신하듯
    어떤 수식에 대입시키기위한 변수의 틀을 미리 만들어 두는 개념
    
'''


# 플레이스 홀더 정의
a = tf.placeholder(tf.int32,[3])

# 배열의 모든 값을 2배 하는 연산 정의
b = tf.constant(2)
x2_op = a * b



sess = tf.Session()


# 플레이스 홀더에 값을 넣고 실행
r1 = sess.run(x2_op, feed_dict={ a:[1,2,3] })
print(r1)

row = [10,20,30]
r2 = sess.run(x2_op, feed_dict={ a:row })
print(r2)










