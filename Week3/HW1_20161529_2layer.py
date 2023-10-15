import numpy as np
import random
import matplotlib.pyplot as plt

epochs = 10000
learning_rate = 0.05

# Input data setting
## XOR data 
## 입력 데이터들, XOR Table에 맞게 정의해놓았습니다.
train_inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_out = np.array([0, 1, 1, 0])


##-----------------------------------##
##------- Activation Function -------##
##-----------------------------------##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and biases
W1_2 = np.random.randn(2, 4)  # First layer weights
b1_2 = np.zeros((1, 4))  # First layer biases
W2_2 = np.random.randn(4, 1)  # Second layer weights
b2_2 = np.zeros((1, 1))  # Second layer biases

errors = []
for epoch in range(epochs):
    # 데이터 4가지 중 랜덤으로 하나 선택
    idx = np.random.randint(4)

    # 입력 데이터 xin과 해당하는 정답 ans 불러오기
    xin = train_inp[idx].reshape(1, 2)
    ans = train_out[idx]

    # Forward Pass
    net1 = sigmoid(np.matmul(xin, W1_2) + b1_2)
    net2 = sigmoid(np.matmul(net1, W2_2) + b2_2)

    # Mean Squared Error (MSE)로 loss 계산
    predictions = net2
    targets = ans
    loss = np.mean((predictions - targets) ** 2)

    # Backpropagation을 통한 Weight의 Gradient calculation(update)
    delta_W2 = np.matmul(net1.T, 2 * (predictions - targets) * (1 - net2**2))
    delta_b2 = 2 * (predictions - targets) * (1 - net2**2)
    delta_W1 = np.matmul(xin.T, np.matmul(2 * (predictions - targets) * (1 - net2**2), W2_2.T) * (1 - net1**2))
    delta_b1 = np.matmul(2 * (predictions - targets) * (1 - net2**2), W2_2.T) * (1 - net1**2)

    # 각 weight의 update 반영
    W2_2 -= learning_rate * delta_W2
    b2_2 -= learning_rate * delta_b2
    W1_2 -= learning_rate * delta_W1
    b1_2 -= learning_rate * delta_b1

    # 500번째 epoch마다 loss를 프린트합니다.
    if epoch % 2000 == 0:
        print("epoch [{}/{}] loss: {:.4f}".format(epoch, epochs, float(loss)))

    # plot을 위해 값 저장
    errors.append(loss)

for idx in range(4):
    xin = train_inp[idx].reshape(1, 2)
    ans = train_out[idx]

    net1 = sigmoid(np.matmul(xin, W1_2) + b1_2)
    net2 = sigmoid(np.matmul(net1, W2_2) + b2_2)

    pred = net2

    print("input: ", xin, ", answer: ", ans, ", pred: {:.4f}".format(float(pred)))


# 학습이 끝난 후, loss를 확인합니다.
loss = np.array(errors)
plt.plot(loss.reshape(epochs))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


np.savetxt("20161529_weight1_layer2.txt", (W1_2), fmt="%s")
np.savetxt("20161529_bias1_layer2.txt", (b1_2), fmt="%s")
np.savetxt("20161529_weight2_layer2.txt", (W2_2), fmt="%s")
np.savetxt("20161529_bias2_layer2.txt", (b2_2), fmt="%s")
