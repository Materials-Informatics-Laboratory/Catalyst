import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import glob
import os

path = str(Path(__file__).parent)
results_dir = os.path.join(path,'results')
loss_file = 'loss.data'

model_dirs = glob.glob(os.path.join(path,results_dir,'*'))

fig, ax = plt.subplots(nrows=1, ncols=len(model_dirs ),sharex=True,sharey=False)
for i,model in enumerate(model_dirs):
    of = open(os.path.join(model,loss_file),'r')
    loss = [[],[]]
    for lines in of:
        line = lines.split()
        if len(line) == 2:
            loss[0].append(float(line[0]))
            loss[1].append(float(line[1]))
    of.close()
    x = np.linspace(0,len(loss[0]) - 1,len(loss[0]))
    ax[i].plot(x,loss[0],linestyle='--',marker='o',color='r',markeredgecolor='k',label='Training')
    ax[i].plot(x, loss[1], linestyle='--', marker='o', color='b', markeredgecolor='k', label='Validation')
    ax[i].set_yscale('log')
    ax[i].set_xlabel('Epoch #')
    ax[i].set_ylabel('Loss')
ax[0].legend(loc="upper right")
plt.show()




