import json
import numpy as np
import matplotlib.pyplot as plt

data = json.load(open("rec.json"))
train_loss = []
test_loss = []
for piece in data:
    train_loss.append(piece[1])
    test_loss.append(piece[2])

x = list(range(len(train_loss)))
ax1 = plt.subplot(1,1,1)

ax1.plot(x, train_loss, color="red",linewidth=1, label = "train loss")
ax1.plot(x, test_loss, color="blue",linewidth=1, label = "test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("train/test loss with respect to epoch")
ax1.legend()
plt.show()