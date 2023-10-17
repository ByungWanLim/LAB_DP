import numpy as np
import random
import matplotlib.pyplot as plt

epochs = 30000
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
W1_2 = np.random.randn(2, 2)  # First layer weights
b1_2 = np.zeros((1, 2))  # First layer biases
W2_2 = np.random.randn(2, 1)  # Second layer weights
b2_2 = np.zeros((1, 1))  # Second layer biases

errors = []
for epoch in range(epochs):
    for i in range(len(train_inp)):
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
        delta_W2 = np.matmul(net1.T, 2 * (predictions - targets) * net2 * (1 - net2))
        delta_b2 = 2 * (predictions - targets) * net2 * (1 - net2)
        delta_W1 = np.matmul(xin.T, np.matmul(2 * (predictions - targets) * net2 * (1 - net2), W2_2.T) * net1 * (1 - net1))
        delta_b1 = np.matmul(2 * (predictions - targets) * net2 * (1 - net2), W2_2.T) * net1 * (1 - net1)

        # 각 weight의 update 반영
        W2_2 -= learning_rate * delta_W2
        b2_2 -= learning_rate * delta_b2
        W1_2 -= learning_rate * delta_W1
        b1_2 -= learning_rate * delta_b1

    # plot을 위해 값 저장
    errors.append(loss)

    # 5000번째 epoch마다 loss를 프린트합니다.
    if epoch % 5000 == 0:
        print("epoch [{}/{}] loss: {:.4f}".format(epoch, epochs, float(loss)))


# 학습이 끝난 후, loss를 확인합니다.
loss = np.array(errors)
plt.plot(loss.reshape(epochs))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# -----------------------------------#
# --------- Testing Step ------------#
# -----------------------------------#


for idx in range(4):
    xin = train_inp[idx].reshape(1, 2)
    ans = train_out[idx]

    net1 = sigmoid(np.matmul(xin, W1_2) + b1_2)
    net2 = sigmoid(np.matmul(net1, W2_2) + b2_2)

    pred = net2

    print("input: ", xin, ", answer: ", ans, ", pred: {:.4f}".format(float(pred)))





np.savetxt("/content/drive/My Drive/Colab Notebooks/20161529_weight_layer2.txt", (W1_2, b1_2, W2_2, b2_2), fmt="%s")

