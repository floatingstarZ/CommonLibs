import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(1,10,20)
y=np.sin(x)*3
error=0.05+0.15*x # 误差范围函数
error_range=np.array([error*0.1,error]) # 下置信度和上置信度
print(error_range)

plt.errorbar(x,y,yerr=error_range,fmt='o:',ecolor='hotpink',
			elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
print(x.shape, y.shape, error.shape)
plt.errorbar(x,y,yerr=error_range,fmt='-o',ecolor=(0, 0, 1),color=(1, 0, 0), elinewidth=2,capsize=4)
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
plt.pause(5)
