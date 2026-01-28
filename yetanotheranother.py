import numpy as np
import pandas as pd

n = np.array([784, 16, 16, 10])
L = len(n)-1

train = pd.read_csv("train.csv")
X = train.iloc[:, 1:].to_numpy().T/255 # inputs
Y = train["label"].to_numpy() # expected outputs

N = X.shape[1] # Number of training examples


print(N)


