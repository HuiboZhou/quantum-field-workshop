import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./root/桌面/pion_gamma15_p0_t0_1.txt') 
cfg = int(len(data) / 64) 
a0 = 0.197/0.12 
m = np.zeros([cfg, 63]) 
M = np.zeros(63) 
Err = np.zeros(63)

for i in range(cfg): 
    for j in range(1,63): 
        m[i,j] = a0 * (np.log(data[i * 64 + j, 7] / data[i * 64 + j + 1, 7]))

for i in range(63): 
    M[i] = np.mean(m[:,i]) 
    Err[i] = np.std(m[:,i])

plt.figure(dpi = 150) 
plt.xlim(0,28) 
plt.errorbar(np.linspace(2,27,26), M[2:-35], yerr = Err[2:-35], fmt = 'o', elinewidth = 2, ms = 5, mfc="c", capsize = 3) plt.plot(np.linspace(0,28,27), 0.3 * np.ones(27),'r--')
plt.plot(np.linspace(0,28,27), 0.3 * np.ones(27),'r--')
plt.show()




