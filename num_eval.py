import torch
import numpy as np
import mnist

from model import Model

x_test, t_test= mnist.load()[2], mnist.load()[3]

model=Model()
model.eval()
ckpt=torch.load("weight_arr_2.pt")
model.load_state_dict(ckpt)


index=5000
success=0
for index in range(10000):
	output=model.forward(torch.Tensor(x_test[index]))
	num=torch.argmax(output).item()
	eval_num[index]=num
	if num==t_test[index]:
		success+=1
print(success/10000 * 100)
ind=0
b=0
val=np.zeros(10000)
while True:
	out=model(torch.Tensor(x_test[b:b+1000]))
	for i in range(b,b+1000):
		x=torch.argmax(out[ind]).item()
		val[i]=x
		ind+=1
	if b+1000==10000:
		break
	else:
		b+=1000
		if ind==1000:
			ind=0
		
for i in range(10000):
	if val[i]==t_test[i]:
		success+=1
print(success/10000 * 100)
