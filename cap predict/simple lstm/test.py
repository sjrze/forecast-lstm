# import library
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read our data from directory
df = pd.read_csv('1.csv')
x = df.Time.values
y = df.No1.values
df = df.iloc[:, 1:2].values

# plot our data
plt.subplots(figsize=(10, 10))

plt.plot(x, y)

plt.title('original cap.no.1')
plt.xlabel('Time')
plt.ylabel('No1')
plt.savefig('foo.png')


# make our train and test based on our data
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


# normalization
sc = MinMaxScaler()
training_data = sc.fit_transform(df)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

# test range for our predicton
train_size = int(len(y) * 0.50)
test_size = len(y) - train_size

# convert data to numpy array
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

# our lstm model
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, ):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


# model hyperparameters
num_epochs = 2000
learning_rate = 0.01
input_size = 1
hidden_size = 2
num_layers = 3
num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, trainY)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# inverse our prediction for plot and show the result
lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)



#  plot our result
plt.subplots(figsize=(5, 5))
plt.axvline(x=train_size, c='b', linestyle='-.', label = 'predict')
plt.plot(dataY_plot, c = 'b')
plt.plot(data_predict, c = 'r', label = 'predict')
plt.xlabel('Time')
plt.ylabel('cap .no1')
plt.suptitle('LSTM prediction on capacity')
plt.legend(['train_size ', 'original rate', ' LSTM prediction'])
# plt.savefig('myimage.svg', format='svg', dpi=1200)
plt.show()
