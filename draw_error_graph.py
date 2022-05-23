import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_file_name = 'results/FashionMNIST_loss.csv'
loss = pd.read_csv(csv_file_name)
train_error = loss['Average Training Error']
validate_error = loss['Average Validation Error']

plt.plot(train_error, c='b', label='train')
plt.plot(validate_error, c='r', label='validate')
plt.title('Average Reconstruction Error on FashionMNIST')
plt.ylabel('Avg. L2 squared error')
plt.xlabel('epoch')
plt.xlim(xmin=0)
plt.legend()
plt.show()