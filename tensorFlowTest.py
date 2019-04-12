import tensorflow as tf

# 텐서에서 사용할 상수 선언
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)


# 연산 정의
calc1 = a + b + c
calc2 = a + b * c
# 바로 출력안됨


# 세션 시작하기
sess = tf.Session()
res1 = sess.run(calc1)
res2 = sess.run(calc2)

print(res1)
print(res2)
