import numpy as np

n = [2, 3, 3, 1]
L = len(n)-1

def prepare_data():
  X = np.array([
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
  ])
  y = np.array([0,1,1,0,0,1,1,0,1,0])
  m = len(X)
  Inputs = X.T
  Y = y.reshape(n[L], m)

  return Inputs, Y

Inputs, Y = prepare_data()

weights = []
biases = []

for i in range(len(n)-1):
    weights.append(np.random.rand(n[i + 1], n[i]) - 0.5)
    biases.append(np.random.rand(n[i+1], 1) - 0.5)



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)




def forward(inputs):
    activations = [inputs]   # A0 = inputs
    zs = []

    for i in range(len(weights)):
        z = weights[i] @ activations[-1] + biases[i]
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)

    return activations, zs

def backward(activations, zs, y):
    m = y.shape[1]

    dW = [None] * len(weights)
    dB = [None] * len(biases)

    # Output layer error
    delta = activations[-1] - y   # (A_L - Y)

    for l in reversed(range(len(weights))):
        dW[l] = (1 / m) * delta @ activations[l].T
        dB[l] = (1 / m) * np.sum(delta, axis=1, keepdims=True)

        if l != 0:
            delta = (weights[l].T @ delta) * sigmoid_derivative(activations[l])

    return dW, dB



def cost(y_hat, y):

    losses = -( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )
    m = y_hat.reshape(-1).shape[0]
    summed_losses = (1 / m) * np.sum(losses, axis=1)

    return np.sum(summed_losses)

def update_params(dW, dB, lr=0.1):
    for i in range(len(weights)):
        weights[i] -= lr * dW[i]
        biases[i] -= lr * dB[i]


epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    activations, zs = forward(Inputs)
    y_hat = activations[-1]

    loss = cost(y_hat, Y)
    dW, dB = backward(activations, zs, Y)
    update_params(dW, dB, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


predictions = (y_hat > 0.5).astype(int)
print("Predictions:", predictions)
print("True labels:", Y)




