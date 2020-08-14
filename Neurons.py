import numpy as np

# 参考网址：https://zhuanlan.zhihu.com/p/59385110

def sigmoid(x):
    # function f(x) = 1 / (1 + e^(-x))
    return 1/(1+np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)


# 损失函数
def mse_loss(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()


class Neuron:
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs)+self.bias
        return sigmoid(total)


class OurNeuralNetwork:
    def __init__(self):
        super().__init__()

        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
        h2 = sigmoid(self.w3*x[0]+self.w4*x[1]+self.b2)

        o1 = sigmoid(self.w5*h1+self.w6*h2+self.b3)

        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                # h2 node
                sum_h2 = self.w3*x[0] + self.w4*x[1] + self.b2
                h2 = sigmoid(sum_h2)

                # o1 node
                sum_o1 = self.w5*h1+self.w6*h2+self.b3
                o1 = sigmoid(sum_o1)

                # 计算导数
                y_pred=o1
                
                d_L_d_ypred=-2*(y_true-y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)



#weights = np.array([0, 1])
#bias = 4
#n = Neuron(weights, bias)

#x = np.array([2, 3])
# print(n.feedforward(x))

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))
