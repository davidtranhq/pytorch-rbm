from train_mnist import train_rbm
from random import uniform

for i in range(30):
    lr = uniform(1e-6, 1e-1)
    decay = uniform(1e-9, 1e-3)
    train_rbm(str(i), 500, 1, lr, 0.5, decay, 30)

