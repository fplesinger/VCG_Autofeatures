#Not required, just to get some insight into features behavior.
#Basic feature analysis for VCG Features project
#Produces images into folder /results
#and textual output (p-values, AUCs).
#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import pline
from scipy.stats import mannwhitneyu
from sklearn import metrics
from colorama import Fore,Back

print("Python version: ")
import sys
print(sys.version)


#pth = "Detected_A_B_89_cases_XYZQRS.pkl"
pth = "results_data/COG_Detected_170_cases-all_XYZQRS_XO.pkl"

print("Loading...")

data = pd.read_pickle(pth)

print("File loaded",data.columns)
print(data.shape)
pline()


print("Removing nans")
data = data.dropna()
print(data.shape)

data=data[data.Measured=="Before"]

clns = data.columns

pline()

minRRs=[]
maxRRs=[]

RRs=[]

sexs=[]

meanRRs = []
stdRRs = []

#add feature min RR
for ri in range(data.shape[0]):
    minRR=np.nan
    maxRR=np.nan
    rrf=[]

    qrsLocs = data.QRS.values[ri]

    if len(qrsLocs>2):
        rrf=np.diff(qrsLocs)

        minRR = min(rrf)
        maxRR = max(rrf)

    minRRs.append(minRR)
    maxRRs.append(maxRR)

    RRs.append(rrf)

    meanRRs.append(np.mean(rrf))
    stdRRs.append(np.std(rrf))

    sexF = 0
    if "ova_" in data.File.values[ri]:
        sexF=1

    sexs.append(sexF)

data["Feas_sexF"]=sexs

data["Feas_minRR"]=minRRs
data["Feas_maxRR"]=maxRRs
data["RRs"]=RRs

data["Feas_meanRR"]=meanRRs
data["Feas_stdRR"]=stdRRs


dataN=data[data.Recidive==0]
dataP=data[data.Recidive==1]



#graphs for RRs
valsN=np.concatenate(dataN.RRs.values)
valsP=np.concatenate(dataP.RRs.values)

plt.figure(figsize=(10,5))
plt.suptitle("RR intervals for responders (N="+str(dataN.shape[0])+") and responders (N="+str(dataP.shape[0]))
plt.subplot(2,1,1)
sns.violinplot(x=valsN, inner=None)
sns.swarmplot(x=valsN,color="white",edgecolor="gray")
plt.grid()
plt.xlim(0,1500)
plt.ylabel("Responders")

plt.subplot(2,1,2)

sns.violinplot(x=valsP, inner=None)
sns.swarmplot(x=valsP,color="white",edgecolor="gray")
plt.xlim(0,1500)
plt.grid()
plt.ylabel("Non-responders")
plt.xlabel("RR-interval (ms)")

plt.savefig("results/RR_swarms.png")
plt.show()

#show graphs for RR in responders/non-responders


for fea in data.columns:
    if fea[:3]!="Fea":
        continue

    pline("-")
    print("Analysing",fea)

    valsN = dataN[fea].values
    valsP = dataP[fea].values

    stats, p = mannwhitneyu(valsN, valsP)

    #pvals.append(p)

    nm = fea.replace("Feas_", "")

    clr = 'lightgray'
    if p <= 0.001:
        clr = 'yellowgreen'
        nm = nm + "\n%.5f" % p + "\n***"
    elif p <= 0.01:
        clr = 'yellowgreen'
        nm = nm + "\n%.3f" % p + "\n**"
        print(Back.GREEN, end="")
    elif p <= 0.05:
        clr = 'orange'
        nm = nm + "\n%.3f" % p + "\n*"
        print(Back.CYAN,end="")




    print("Stats:",stats," p-value:%.3f"%p)

    print("Means: N:%.2f"%np.mean(valsN)+" P:%.2f"%np.mean(valsP))


    fpr, tpr, thresholds = metrics.roc_curve(data.Recidive, data[fea], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    if auc<0.5:auc = 1-auc

    print("AUC:%.3f"%auc,Back.RESET)


    if p<0.001:
        plt.figure(figsize=(20,8))
        plt.subplot(1,2,1)
        plt.boxplot([valsN,valsP],showfliers=False)

        #if "expPAD" in fea:
        #plt.ylim(0,np.percentile(data[fea].values,95))

        #if "PADsum" in fea:
        #    plt.ylim(0,1200)

        plt.xticks([1,2],["Responders","Non-responders"])
        plt.grid()
        axes = plt.gca()
        ylims=axes.get_ylim()

        plt.subplot(1,2,2)
        sns.histplot(data=data, x=fea, hue="Recidive",binrange=ylims)

        plt.suptitle(fea+" | p=%.3f"%p+" | AUC=%.2f"%auc)

        plt.savefig("results/"+fea+"_boxplot.png")
        plt.show()




    #print("Gender - specific stats:",stats," pm:%.5f"%pm," pf:%.5f"%pf)
