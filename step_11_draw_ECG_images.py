#It is not required
#This just prints ECG data in separate leads
#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tools import pline

from sklearn import metrics
from scipy.stats import mannwhitneyu


print("Reading main dataset with outcomes")
dnm = "results_data/COG_Detected_170_cases-all_XYZQRS_XO_KDE.pkl"
data = pd.read_pickle(dnm)
data=data[data.Measured=="Before"]
print("Done (with reduction). ",data.columns)
print("Shape:",data.shape)


ab = data.avgBeat.values

for ri in range(data.shape[0]):
    print("generating image",ri)

    plt.figure(figsize=(10,14))


    xt = np.arange(0,300,10)
    xl = [str(x-150) for x in xt]

    for v in range(6):
        sh = ab[ri]

        plt.subplot(6,1,v+1)
        plt.plot(sh[450:750,6+v])
        plt.xticks(xt,xl)
        plt.ylabel("V"+str(v+1))
        plt.ylim(-250,250)
        plt.grid()

    plt.suptitle("File "+data.File.values[ri])

    plt.tight_layout()
    fln = "figs\ECGV16_"+ data.File.values[ri].replace(".csv",".png")
    plt.savefig(fln)
    #plt.show()

    plt.close()