import json
import numpy as np
import matplotlib.pyplot as plt

# data = json.load(open("rec.json"))
# train_loss = []
# test_loss = []
# for piece in data:
#     train_loss.append(piece[1])
#     test_loss.append(piece[2])

# x = list(range(len(train_loss)))


data_rnn = json.load(open("rec_rnn.json"))
test_rnn = []
data_lstm = json.load(open("rec_lstm.json"))
test_lstm = []
data_gru = json.load(open("rec_gru.json"))
test_gru = []

for i in range(len(data_gru)):
    test_gru.append(data_gru[i][2])
    test_lstm.append(data_lstm[i][2])
    test_rnn.append(data_rnn[i][2])
x = list(range(len(test_gru)))
ax1 = plt.subplot(1,1,1)

ax1.plot(x, test_gru, color="red",linewidth=1, label = "gru")
ax1.plot(x, test_lstm, color="blue",linewidth=1, label = "lstm")
ax1.plot(x, test_rnn, color="green",linewidth=1, label = "gru")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("rnn,lstm,gru loss with respect to epoch")
ax1.legend()
# plt.show()
plt.savefig("rnn_lstm_gru_loss.png")