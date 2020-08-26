import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt('/root/桌面/pion_gamma15_p0_t0_1.txt') 
cfg = int(len(data) / 64) 
print(cfg)
a0 = 0.197/0.12 
m = np.zeros([cfg, 63]) 
m2 = np.zeros([cfg, 62])
ci = np.zeros([cfg, 63])
M = np.zeros(63) 
M_resamp = np.zeros(63)
M_Err = np.zeros(63)
M_resamp_Err = np.zeros(63)



for i in range(cfg): 
    for j in range(0,63): 
        m[i,j] = a0 * (np.log(data[i * 64 + j, 7] / data[i * 64 + j + 1, 7]))
        ci[i,j] = data[i*64+j,7]

#Jacknife resample
c2_resamp = np.zeros_like(ci)
for i in range(ci.shape[0]):
    cols_extracted = np.concatenate((ci[:i],ci[i+1:]),axis=0)
    c2_resamp[i,:]=np.average(cols_extracted,axis=0)
    
for i in range(23):
     M[i] = np.mean(m[:,i])
     M_Err[i]=np.std(m[:,i])/(math.sqrt(cfg-1))
        
# 1.Jm:efective mass  
for i in range(cfg): 
    for j in range(0,62): 
        m2[i,j] = a0 * (np.log(c2_resamp[i, j] / c2_resamp[i,j + 1]))
for i in range(23):
     M_resamp[i] = np.mean(m2[:,i])
     M_resamp_Err[i]=np.std(m2[:,i])/(math.sqrt(cfg-1))*(cfg-1)/math.sqrt(cfg)



plt.figure(dpi = 150) 
plt.xlim(0,22) 
plt.xlabel('Time/Lattice Unit')
plt.ylabel('C2')
plt.title('2-point correlation of Configure mass')
plt.errorbar(np.linspace(2,23,22),M[2:-39],yerr=M_Err[2:-39],fmt='o:',elinewidth=2,ms=5,mfc="c",capsize=3,label='original')
plt.errorbar(np.linspace(2,23,22),M_resamp[2:-39],yerr=M_resamp_Err[2:-39],fmt='s',elinewidth=2,ms=5,mfc="c",label='Jacknife resample')
plt.plot(np.linspace(0,23,22), 0.31 * np.ones(22),'r--')
plt.show()


def func(x, a, b, c):
    return a * np.exp(-b * x) + c
xdata=np.linspace(2,23,22)
ydata=M_resamp[2:-39]
plt.plot(xdata,ydata,'b-')
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, ydata, 'b-')
y2 = [func(i, popt[0],popt[1],popt[2]) for i in xdata]
plt.plot(xdata,y2,'r--')
print(popt)
plt.show()



