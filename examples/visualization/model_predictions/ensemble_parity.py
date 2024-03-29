import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp

from pathlib import Path
import glob
import os

path = str(Path(__file__).parent)
results_dir = os.path.join(path,'results')
sub_types = ['binary','ternary']
pred_file = 'all_indv_pred.data'

y_correction = 10.0

fig, ax = plt.subplots(nrows=1, ncols=len(sub_types),sharex=True,sharey=True)

for s,st in enumerate(sub_types):
    n_models = glob.glob(os.path.join(results_dir,st,'*'))
    pred = [[],[]]
    for i,n in enumerate(n_models):
        of = open(os.path.join(results_dir,st,n,pred_file),'r')
        for lines in of:
            line = lines.split()
            if len(line) == 2:
                ty = float(line[0]) - y_correction
                py = float(line[1]) - y_correction
                if i == 0:
                    pred[0].append([ty])
                    pred[1].append([py])
                else:
                    for j in range(len(pred[0])):
                        if ty == pred[0][j][0]:
                            pred[0][j].append(ty)
                            pred[1][j].append(py)
                            break
        of.close()

    final_dft_vals = []
    avg_ml_vals = []
    std_vals = []

    for i,ml in enumerate(pred[1]):
        avg_ml_vals.append(np.mean(np.array(ml)))
        std_vals.append(np.std(np.array(ml)))
        final_dft_vals.append(pred[0][i][0])

    if len(sub_types) > 1:
        ax[s].errorbar(final_dft_vals, avg_ml_vals, yerr = std_vals, fmt ='',linestyle='',color='r',alpha=.5)
        ax[s].scatter(final_dft_vals, avg_ml_vals,facecolor='white',edgecolor='k')
        ax[s].plot(final_dft_vals,final_dft_vals,'k')
        ax[s].set_xlabel('DFT Enthalpy (eV/atom)')
        ax[s].set_ylabel('GNN Enthalpy (eV/atom)')
    else:
        ax.errorbar(final_dft_vals, avg_ml_vals, yerr=std_vals, fmt='', linestyle='', color='r', alpha=.5)
        ax.scatter(final_dft_vals, avg_ml_vals, facecolor='white', edgecolor='k')
        ax.plot(final_dft_vals, final_dft_vals, 'k')
        ax.set_xlabel('DFT Enthalpy (eV/atom)')
        ax.set_ylabel('GNN Enthalpy (eV/atom)')

    mae = mean_absolute_error(final_dft_vals, avg_ml_vals,)
    rmse = mean_squared_error(final_dft_vals, avg_ml_vals, squared=False)
    mse = mean_squared_error(final_dft_vals, avg_ml_vals, squared=True)
    r2 = r2_score(final_dft_vals, avg_ml_vals,)
    ks = ks_2samp(final_dft_vals, avg_ml_vals,)

    miss = 0
    for i,ml in enumerate(avg_ml_vals):
        if final_dft_vals[i] < 0.0 and ml > 0.0:
            miss += 1
    miss_rate = float(miss) / float(len(avg_ml_vals))

    scale = 1000.0
    print('MAE (meV/atom) = ' + str(mae*scale))
    print('RMSE (meV/atom) = ' + str(rmse*scale))
    print('MSE (meV/atom) = ' + str(mse*scale))
    print('KS = ' + str(ks))
    print('r2 = ' + str(r2))
    print('miss rate ' + str(miss_rate))
    print('\n\n')

plt.show()




