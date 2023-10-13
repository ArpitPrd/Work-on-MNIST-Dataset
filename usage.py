import mnist
import numpy as np
from PIL import Image
import torch
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
	
def f():
	change()
	L = torch.nn.Linear(784, 10)
	s = torch.nn.Softmax(10)
	model2=torch.nn.Sequential(L)
	loss_fn = torch.nn.CrossEntropyLoss()
	learning_rate=0.001
	opt = torch.optim.SGD(list(model2.parameters()), lr=0.001)
	for i in range(60000):
		opt.zero_grad()
		inp=torch.Tensor(x_train[i])
		target=torch.Tensor(t_train[i])
		loss=loss_fn(model2(inp),target)
		loss.backward()
		opt.step()
	torch.save(model2.state_dict(),"weight_arr.pt")

def g():
	change()
	L = torch.nn.Linear(784, 10)
	model=torch.nn.Sequential(L)
	loss_fn = torch.nn.CrossEntropyLoss()
	learning_rate=0.001
	opt = torch.optim.SGD(list(model.parameters()), lr=0.001)
	b=0
	count=0
	batchsize=1000
	while True:
		opt.zero_grad()
		inp=torch.Tensor(x_train[b:b+batchsize])
		target=torch.Tensor(t_train[b:b+batchsize])
		loss=loss_fn(model(inp), target)
		loss.backward()
		opt.step()
		print(loss.item())
		if count > 100:
			break
		else:
			count+=1
			if b+batchsize==60000:
				b=0
			else:
				b+=batchsize
	torch.save(model.state_dict(),"weight_arr.pt")




def h()
	change()
	batchsize=1000
	con=torch.nn.Conv2d(in_channels=batchsize, out_channels=16*batchsize, kernel_size=(3,3))
	L = torch.nn.Linear(784, 10)
	avg=torch.nn.AdaptiveAvgPool2d(output_size=(batchsize,28, 28))
	reshape1=torch.reshape(shape=(batchsize,28,28))
	reshape2=torch.reshape(shape=(batchsize,784))
	model=torch.nn.Sequential(reshape1,con,avg,reshape2,L)
	loss_fn = torch.nn.CrossEntropyLoss()
	learning_rate=0.001
	opt = torch.optim.SGD(list(model.parameters()), lr=0.001)
	b=0
	count=0
	while True:
		opt.zero_grad()
		inp=torch.Tensor(x_train[b:b+batchsize])
		target=torch.Tensor(t_train[b:b+batchsize])
		loss=loss_fn(model(inp), target)
		loss.backward()
		opt.step()
		if count > 100:
			break
		else:
			count+=1
			if b+batchsize==60000:
				b=0
			else:
				b+=batchsize
	torch.save(model.state_dict(),"weight_arr_1.pt")	

h()
