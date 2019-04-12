import pandas as pd
import numpy as np
import tensorflow as tf

csv = pd.read_csv('bmi.csv')
# print(csv.head())

'''
   height  weight   label
0     142      62     fat
1     142      73     fat
2     177      61  normal
3     187      48    thin
4     153      60     fat

'''

# 정규화
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

# 레이블을 배열로 변환
bclass = { 'thin':[1,0,0],'normal':[0,1,0],'fat':[0,0,1] }
csv['label_pat'] = csv['label'].apply(lambda x : np.array(bclass[x]))

# print(csv.head())


# 테스트를 위한 데이터 분류
test_csv = csv[15000:20000]

test_pat = test_csv[["weight","height"]]
test_ans = list(test_csv["label_pat"])

# print(test_pat.head())
# print(test_ans)
# print(type(test_ans))

# 플레이스 홀더 만들기
# 훈련시키기위한 문제
x = tf.placeholder(tf.float32,[None, 2])

# 훈련시키기위한 답
y_ = tf.placeholder(tf.float32,[None, 3])


# 변수 선언하기
#                       (feature의수, 답의 수)
w = tf.Variable(tf.zeros([2,3])) # 가중치
b = tf.Variable(tf.zeros([3]))  # 바이어스


# 소프트맥스 회귀 정의
y = tf.nn.softmax(tf.matmul(x,w) + b)


# 모델 훈련하기
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)


# 정답률구하기
pridict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(pridict,tf.float32))

# 세션 시작하기
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 학습시키기
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 +i : 1 + i + 100]
    x_pat = rows[["weight","height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_:y_ans}
    sess.run(train,feed_dict=fd)

    if step % 500 == 0 :
        cre = sess.run(cross_entropy,feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_:test_ans})
        print(step,"//",cre,"//",acc)

# 최종적인 정답률
acc = sess.run(accuracy, feed_dict={x: test_pat, y_:test_ans})
print(acc)



sess.close()











