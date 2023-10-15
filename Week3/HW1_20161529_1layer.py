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
W1_1 = np.random.randn(2, 1)  # First layer weights
b1_1 = np.zeros((1, 1))  # First layer biases

errors = []
for epoch in range(epochs):
    # 데이터 4가지 중 랜덤으로 하나 선택
    idx = np.random.randint(4)

    # 입력 데이터 xin과 해당하는 정답 ans 불러오기
    xin = train_inp[idx].reshape(1, 2)
    ans = train_out[idx]

    # Forward Pass
    net1 = sigmoid(np.matmul(xin, W1_1) + b1_1)

    # Mean Squared Error (MSE)로 loss 계산
    predictions = net1
    targets = ans
    loss = np.mean((predictions - targets) ** 2)

    # Backpropagation을 통한 Weight의 Gradient calculation(update)
    delta_W1 = np.matmul(xin.T, 2 * (predictions - targets) * (1 - net1**2))
    delta_b1 = 2 * (predictions - targets) * (1 - net1**2)

    # 각 weight의 update 반영
    W1_1 -= learning_rate * delta_W1
    b1_1 -= learning_rate * delta_b1

    # 500번째 epoch마다 loss를 프린트합니다.
    if epoch % 2000 == 0:
        print("epoch [{}/{}] loss: {:.4f}".format(epoch, epochs, float(loss)))

    # plot을 위해 값 저장
    errors.append(loss)

for idx in range(4):
    xin = train_inp[idx].reshape(1, 2)
    ans = train_out[idx]

    net1 = sigmoid(np.matmul(xin, W1_1) + b1_1)

    pred = net1.item() 

    print("input: ", xin, ", answer: ", ans, ", pred: {:.4f}".format(float(pred)))

# 학습이 끝난 후, loss를 확인합니다.
loss = np.array(errors)
plt.plot(loss.reshape(epochs))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

np.savetxt("20161529_weight_layer1.txt", (W1_1), fmt="%s")
np.savetxt("20161529_bias_layer1.txt", (b1_1), fmt="%s")