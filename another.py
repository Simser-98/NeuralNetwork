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



def forward(inputs):

    outputs = weights[0] @ inputs + biases[0]

    for i in range(len(n) - 2):
        '''print("W: ", weights[i + 1].shape)
        print("B: ", biases[i + 1].shape)
        print("Z: ", outputs.shape)'''

        Activated_outputs = sigmoid(outputs)
        outputs = weights[i + 1] @ Activated_outputs + biases[i + 1]

    return sigmoid(outputs)

neural_nets_output = forward(Inputs) # neural nets prediction/ y_hat
print("neural net's pass output: ", neural_nets_output)

def cost(y_hat, y):

    losses = -( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )
    m = y_hat.reshape(-1).shape[0]
    summed_losses = (1 / m) * np.sum(losses, axis=1)

    return np.sum(summed_losses)



