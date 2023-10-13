import mnist
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(1, 16, 3, padding=1),
			nn.MaxPool2d((2, 2)),
			nn.ReLU(inplace=True),

			nn.Conv2d(16, 32, 3, padding=2),
			nn.MaxPool2d((2, 2)),
			nn.ReLU(inplace=True),

			#nn.Conv2d(32, 64, 3, padding=2),
			#nn.MaxPool2d((2, 2)),
			#nn.ReLU(inplace=True),

			#nn.AdaptiveAvgPool2d(output_size=1),
			nn.Flatten(),

			nn.Linear(2048, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(128, 10)
		)

	def forward(self, inp):
		batchsize = inp.shape[0]
		inp=torch.reshape(inp, shape=(batchsize, 1, 28, 28))
		inp = self.net(inp)
		return inp

x_train, t_train_num, x_test, t_test_num = mnist.load()

t_train = np.zeros((60000,10))
t_test=np.zeros((60000,10))

def change():
	index = 0
	for i in t_train_num:
		t_train[index, i] = 1
		index+=1
	index=0
	for i in t_test_num:
		t_train[index, i]=1
		index+=1


def h():
	change()
	model=Model()
	opt=torch.optim.SGD(list(model.parameters()), lr=0.001)
	loss_fn = torch.nn.CrossEntropyLoss()
	b=0
	count=0
	batchsize=1000
	while True:
		opt.zero_grad()
		inp=torch.Tensor(x_train[b:b+batchsize])
		y_pred=model(inp)
		y_true=t_train[b:b+batchsize]
		y_true=torch.Tensor(y_true)
		loss=loss_fn(y_pred, y_true)
		loss.backward()
		opt.step()
		if count>100:
			break
		else:
			count+=1
			if b+batchsize==60000:
				b=0
			else:
				b+=batchsize
	torch.save(model.state_dict(), "weight_arr_2.pt")
h()	
