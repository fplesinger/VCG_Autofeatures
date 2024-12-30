#Create KDE maps from CV rounds in VCG Features project
#This script iterartes through each feature.
#It generates KDE maps showing diversity in CV rounds.
#Generated images are exported into /figs_maps

#Finaly, this script generates tables with KDE results

#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23


import numpy as np
import pandas as pd
from colorama import Fore,Back
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy.stats import mannwhitneyu
from scipy.stats import norm

#based on original step 15

file_d="results_data/CV5_autofeatures_Data_85_cases_SMOOTH_median10.pkl"

flnm = file_d

#nonAFstring = ['AVB', 'KES', 'KES kvadrigeminie', 'SR']

def pline(chr="–", num=80):
    print(chr * num)

print("Loading feature dataframe...")
data = pd.read_pickle(flnm)
print("Done")
print("Columns:",data.columns)

resultFrame = pd.DataFrame()



fileAreas = "results_data/COG_expPAreas_offset_-05_06_220_lng_001_05_98.npy"
fileSums = "results_data/COG_expPADsums_offset_-05_06_220_lng_001_05_98.npy"
fileXSum = "results_data/COG_xSums_offset_-05_06_220_lng_001_05_98.npy"
filedXMean = "results_data/COG_dXmeans_offset_-05_06_220_lng_001_05_98.npy"
fileYSum = "results_data/COG_ySums_offset_-05_06_220_lng_001_05_98.npy"
filedYMean = "results_data/COG_dYmeans_offset_-05_06_220_lng_001_05_98.npy"
fileZSum = "results_data/COG_zSums_offset_-05_06_220_lng_001_05_98.npy"
filedZMean = "results_data/COG_dZmeans_offset_-05_06_220_lng_001_05_98.npy"

fileXamps = "results_data/COG_xAmps_offset_-05_06_220_lng_001_05_98.npy"
fileYamps = "results_data/COG_yAmps_offset_-05_06_220_lng_001_05_98.npy"
fileZamps = "results_data/COG_zAmps_offset_-05_06_220_lng_001_05_98.npy"

fileexpPADXs = "results_data/COG_expPADXs_offset_-05_06_220_lng_001_05_98.npy"
fileexpPADYs = "results_data/COG_expPADYs_offset_-05_06_220_lng_001_05_98.npy"
fileexpPADZs = "results_data/COG_expPADZs_offset_-05_06_220_lng_001_05_98.npy"

fileYsb0Name = "results_data/COG_ysb0_offset_-05_06_220_lng_001_05_98.npy"

strs = fileAreas.split("_")
offMin = float(strs[4]) / 10
offMax = float(strs[5]) / 10
lngMin = float(strs[8]) / 100  # tady bylo 10
lngMax = float(strs[9]) / 10

offStepsIm = float(strs[6])
offInc = (offMax - offMin) / offStepsIm
#offLbs = ["%.2f" % f for f in np.arange(offMin, offMax, offInc)]

lngStepsIm = float(strs[10].replace(".npy", ""))
lngInc = (lngMax - lngMin) / lngStepsIm
#lngLbs = ["%.2f" % f for f in np.arange(lngMin, lngMax, lngInc)]

imH = offStepsIm/15
imW = lngStepsIm/10
#diagonala
al0 = [(0.01-lngMin)/lngInc,(0.01-offMin)/offInc]
al1 = [(0.5-lngMin)/lngInc,(0.5-offMin)/offInc]

#horizontala
zl0 = [(0.01-lngMin)/lngInc,(0.0-offMin)/offInc]
zl1 = [(0.5-lngMin)/lngInc,(0.0-offMin)/offInc]


print("Loading feature maps")
map_expPArea = np.load(fileAreas)
map_expPADarea = np.load(fileSums)
map_xSum = np.load(fileXSum)
map_dXMean = np.load(filedXMean)
map_ySum = np.load(fileYSum)
map_dYMean = np.load(filedYMean)
map_zSum = np.load(fileZSum)
map_dZMean = np.load(filedZMean)

map_xAmps = np.load(fileXamps)
map_yAmps = np.load(fileYamps)
map_zAmps = np.load(fileZamps)

map_expPADx = np.load(fileexpPADXs)
map_expPADy = np.load(fileexpPADYs)
map_expPADz = np.load(fileexpPADZs)
 
map_ysb0 = np.load(fileYsb0Name)

print("Done")
pline()

print("Loading outcome")
dnm = "results_data/COG_Detected_170_cases-all_XYZQRS_XO.pkl"
dataAllFeas = pd.read_pickle(dnm)
dataAllFeas=dataAllFeas[dataAllFeas.Measured=="Before"]


outcome = dataAllFeas.Recidive==1
RSP = outcome == True
NRSP = outcome == False

print("Done. Columns N=", dataAllFeas.shape[1])
print(dataAllFeas.columns)


pline("█")
print("Analyzing features")
pline()

resultFrame["Name"]=data.Name.values




for c in data.columns:
    if c=="Name":
        continue
    print("Feature ",c)
    pline()

    means = []
    smods = []



    for ri in range(data.shape[0]):
        vals = data[c].values[ri]
        meanv = np.mean(vals)
        smod = np.std(vals)
        means.append(meanv)
        smods.append(smod)

    resultFrame[c+" mean"]=means
    resultFrame[c+" std"]=smods



pline("█")
print("Searching best pos through KDE")
pline()

glob_auc=[]
glob_p=[]

kdeOff=[]
kdeDur=[]
kdeOffRow=[]
kdeDurCol=[]

sqr2pi = np.sqrt(2*np.pi)

if 1==1:

    # vyšetření nejlepší polohy přes KDE
    for r in range(resultFrame.shape[0]):

        pline()
        feaname = resultFrame.Name.values[r]
        testF1 = resultFrame["F1_Test mean"].values[r]
        print("Feature", feaname, "Test F1 (CV5):%.2f" % testF1)

        if not "Feas_" in feaname:


            offsets = data.Offsets.values[r]
            durations = data.Durations.values[r]

            print("Searching KDE maxima...",end="")

            dmp = np.zeros((map_expPArea.shape[1],map_expPArea.shape[2]))

            for i in range(len(offsets)):
                off = offsets[i]
                lng = durations[i]
                rmax = int(round((off - offMin) / offInc))
                cmax = int(round((lng - lngMin) / lngInc))

                for mr in range(map_expPArea.shape[1]):
                    for mc in range(map_expPArea.shape[2]):
                        dR = rmax-mr
                        dC = cmax-mc
                        dist = np.sqrt(dR*dR+dC*dC)

                        dist /=10

                        gs = (np.exp(-(dist*dist)/2))/sqr2pi

                        dmp[mr,mc] +=gs



            kr,kc = np.unravel_index(np.argmax(dmp),dmp.shape)

            kdeOffRow.append(kr)
            kdeDurCol.append(kc)

            rMax = kr*offInc+offMin
            lMax = kc*lngInc+lngMin

            kdeOff.append(rMax)
            kdeDur.append(lMax)

            print("found at map row",kr,"col",kc," Offset%.2f"%rMax," Duration %.2f"%lMax)

            plt.figure()
            plt.figure(figsize=(imW, imH))

            ax = plt.subplot(1, 1, 1)

            ax = sns.heatmap(dmp, xticklabels=10, yticklabels=10)



            #ax = sns.heatmap(dtAreasAUC,xticklabels=10,yticklabels=10,center=0.5)

            plt.title(feaname + " locations CV")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Offset (sec before QRS annotation mark)")

            ax.plot([al0[0], al1[0]], [al0[1], al1[1]], "k:")
            ax.plot([zl0[0], zl1[0]], [zl0[1], zl1[1]], "k:")



            for i in range(len(offsets)):
                off = offsets[i]
                lng = durations[i]

                rmax = int(round((off - offMin) / offInc))
                cmax = int(round((lng - lngMin) / lngInc))
                #ax.scatter(cmax, rmax, marker='o', s=100, color='w')
                ax.scatter(cmax, rmax, marker='4', s=200, color='b')


            #ax.scatter(kc, kr, marker='o', s=200, color='w') #blocked due to paper image
            #ax.scatter(kc, kr, marker='4', s=400, color='k')
            plt.suptitle(flnm)
            plt.savefig("figs_maps/KDE_"+feaname+".png")
            plt.show()
        else:
            kdeOff.append(np.nan)
            kdeDur.append(np.nan)
            kdeOffRow.append(np.nan)
            kdeDurCol.append(np.nan)

pline("█")
print("Evaluating p-values")
pline()

resultFrame["KDE offset"]=kdeOff
resultFrame["KDE duration"]=kdeDur
resultFrame["KDE offset row"]=kdeOffRow
resultFrame["KDE duration col"]=kdeDurCol

#if "113" in flnm:
if 1==1:
    for r in range(resultFrame.shape[0]):

        pline()
        feaname = resultFrame.Name.values[r]
        testF1 = resultFrame["F1_Test mean"].values[r]
        print("Feature",feaname,"Test F1 (CV5):%.2f"%testF1)

        feaname = feaname.replace("Fes_","Feas_")
        if not "Feas_" in feaname:
            meanOff = resultFrame["Offsets mean"].values[r]
            meanLng = resultFrame["Durations mean"].values[r]

            meanOff = resultFrame["KDE offset"].values[r]
            meanLng = resultFrame["KDE duration"].values[r]

            print("KDE Offset:%.4f"%meanOff," Duration:%.4f"%meanLng)

            # vyšetření p-hodnoty
            rmax =int(round((meanOff - offMin) / offInc))
            cmax =int(round((meanLng - lngMin) / lngInc))

            print("Max R:", rmax, " C:", cmax)

            varname = "map_"+feaname
            mp = globals()[varname]
            #mp = mp[nonFisCase]
            #print("Variable ",varname,":",mp.shape)

            if 1==1:
                print("Adding to global dataset")
                dataAllFeas["Feas_KDE_"+feaname]=mp[:,rmax,cmax]

            valsRSP = mp[RSP, rmax, cmax]
            valsNRSP = mp[NRSP, rmax, cmax]

            #removing nans
            valsRSP = valsRSP[~np.isnan(valsRSP)]
            valsNRSP = valsNRSP[~np.isnan(valsNRSP)]

            rspMean = np.mean(valsRSP)
            nrspMean = np.mean(valsNRSP)
        else:
            vals = dataAllFeas[feaname].values
            valsRSP = vals[RSP]
            valsNRSP=vals[NRSP]


        wholeCol = np.hstack([valsRSP, valsNRSP]) * 1
        labels = np.hstack([np.ones(len(valsRSP)), np.zeros(len(valsNRSP))])
        try:
            fpr, tpr, thresholds = metrics.roc_curve(labels, wholeCol, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            stats, p = mannwhitneyu(valsRSP, valsNRSP)

        except:
            print(Back.RED,"Error",Back.RESET)
            auc = 0.5
            p = 1

        if auc<0.5:
            auc = 1-auc

        if p<0.05:
            print(Back.CYAN,end="")
        if p<0.001:
            print(Back.GREEN,end="")

        print("Global P-value (Mann-Whithney U test) %.5f"%p)
        print("Global AUC:%.2f"%auc, Back.RESET)

        glob_auc.append(auc)
        glob_p.append(p)

if len(glob_p)==resultFrame.shape[0]:
    resultFrame["p (KDE)"]=glob_p
    resultFrame["AUC (KDE)"]=glob_auc

pline("=")


resName =  flnm.replace(".pkl","_resTable.xls")
#print("Exporting to excel:",resName)
#resultFrame.to_excel(resName)
#print("Done")

resName = resName.replace(".xls",".pkl")
print("Exporting to pkl",resName)
resultFrame.to_pickle(resName)
print("Done")

dnm = dnm.replace("85",str(dataAllFeas.shape[0]))

enm = dnm.replace(".pkl","_KDE.pkl")

print("Exporting to ",enm)
dataAllFeas.to_pickle(enm)
print("Done.")