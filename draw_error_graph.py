import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

csv_file_name = args.path
loss = pd.read_csv(csv_file_name)
train_error = loss['Average Training Error']
validate_error = loss['Average Validation Error']

plt.plot(train_error, c='b', label='train')
plt.plot(validate_error, c='r', label='validate')
plt.title(f'Average Reconstruction Error from {csv_file_name}')
plt.ylabel('Avg. L1 norm error')
plt.xlabel('epoch')
plt.xlim(xmin=0)
plt.legend()
plt.show()