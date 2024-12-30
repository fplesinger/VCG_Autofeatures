#Produces matrices for VCG Features project
#This script generates matrices for each window (offset+duration) for each feature.

#Input is a single pickle file wih averaged QRS complexes converted to VCG, subjects as rows
#(column "XYZQRSshape", see line 110 for structure)

#Results are saved as 3D-Numpy matrices in a folder /results_data
#Dimensions in produced matrices are organized as follows:
# D1: subject ID
# D2: Window start
# D3: window duration

#Warning - takes a long time to proceed (approx 10 minutes)
#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import signal_proc
from tools import pline


pth = "results_data/COG_Detected_170_cases-all_XYZQRS.pkl"

print("Loading...")

data = pd.read_pickle(pth)

print("File loaded",data.columns)
data=data[data.Measured=="Before"]

rows = data.shape[0]
print("Data rows:",rows)
pline()





#durations=[]
#areas=[]
#expAdSums=[]
#expPAreas=[]

offsetSteps = 220 #100 200
offsetMin = -0.5 #0.1 #mínus je doprava za QRS smerem do T
offsetMax = 0.6
offsetInc = (offsetMax-offsetMin)/offsetSteps

lngSteps = 98  #80
lngMin = 0.01  #0.1
lngMax = 0.5
lngInc = (lngMax-lngMin)/lngSteps

expPAreas = np.zeros((rows,offsetSteps,lngSteps))
expPAreas[:]=np.nan

expPADsums = np.zeros((rows,offsetSteps,lngSteps))
expPADsums[:]=np.nan

Xsums = np.zeros((rows,offsetSteps,lngSteps))
Xsums[:]=np.nan
dXmeans = np.zeros((rows,offsetSteps,lngSteps))
dXmeans[:]=np.nan

Ysums = np.zeros((rows,offsetSteps,lngSteps))
Ysums[:]=np.nan
dYmeans = np.zeros((rows,offsetSteps,lngSteps))
dYmeans[:]=np.nan

Zsums = np.zeros((rows,offsetSteps,lngSteps))
Zsums[:]=np.nan
dZmeans = np.zeros((rows,offsetSteps,lngSteps))
dZmeans[:]=np.nan

Xamps = np.zeros((rows,offsetSteps,lngSteps))
Xamps[:]=np.nan

Yamps = np.zeros((rows,offsetSteps,lngSteps))
Yamps[:]=np.nan

Zamps = np.zeros((rows,offsetSteps,lngSteps))
Zamps[:]=np.nan

expPADXs = np.zeros((rows,offsetSteps,lngSteps))
expPADXs[:]=np.nan

expPADYs = np.zeros((rows,offsetSteps,lngSteps))
expPADYs[:]=np.nan

expPADZs = np.zeros((rows,offsetSteps,lngSteps))
expPADZs[:]=np.nan

Ysb0s = np.zeros((rows,offsetSteps,lngSteps))
Ysb0s[:]=np.nan

for r in range(rows):
    print("Row",r)

    fs = 1000

    #sr = int(fs*0.2)

    xyz = data.XYZQRSshape.values[r]

    X = xyz[0, :]
    Y = xyz[1, :]
    Z = xyz[2, :]

    dX = np.diff(X)
    dY = np.diff(Y)
    dZ = np.diff(Z)


    for off in np.arange(offsetMin, offsetMax, offsetInc):
        offID =int(round((off - offsetMin) / offsetInc))

        for lng in np.arange(lngMin, lngMax, lngInc):
            lngID = int(round((lng-lngMin)/lngInc))

            pExpStartOffset = int(fs*off) #default 0.275; max 1200 samples = 0.6sec, min 0.1sec
            pExpLng = int(fs*lng) #default 0.2sec; max 0.5sec; min 0.1sec



            lng = len(X)

            #samplesFrom = int(lng/2-sr)
            #sampleTo = int(lng/2+sr)

            pSampleFrom = int(lng/2-pExpStartOffset) #nalevo od stredu
            pSampleTo = pSampleFrom+pExpLng

            Xp = X[pSampleFrom:pSampleTo]
            Yp = Y[pSampleFrom:pSampleTo]
            Zp = Z[pSampleFrom:pSampleTo]

            adXp = np.sum(np.abs(np.diff(Xp)))
            adYp = np.sum(np.abs(np.diff(Yp)))
            adZp = np.sum(np.abs(np.diff(Zp)))

            dX = np.diff(Xp)
            dY = np.diff(Yp)
            dZ = np.diff(Zp)


            expPADXs[r,offID,lngID]   = adXp
            expPADYs[r, offID, lngID] = adYp
            expPADZs[r, offID, lngID] = adZp

            expPADSum=np.sqrt(adXp*adXp+adYp*adYp+adZp*adZp)

            expPADsums[r,offID,lngID]=expPADSum

            #if lngID==68:
            #    print()

            Xamps[r,offID,lngID]=max(Xp)-min(Xp)
            Yamps[r,offID,lngID]=max(Yp)-min(Yp)
            Zamps[r, offID, lngID] = max(Zp) - min(Zp)

            YpsBelowZero = Yp[Yp<0]

            YAreaBZ = 0
            if len(YpsBelowZero>0):
                YAreaBZ = sum(YpsBelowZero)

            Ysb0s[r, offID, lngID] = YAreaBZ

            Xp=np.sum(np.abs(Xp-Xp[0]))
            Yp=np.sum(np.abs(Yp-Yp[0]))
            Zp=np.sum(np.abs(Zp-Zp[0]))

            expPArea=np.sqrt(Xp*Xp+Yp*Yp+Zp*Zp)
            expPAreas[r,offID,lngID] = expPArea

            Xsums[r,offID,lngID]=Xp
            dXmeans[r,offID,lngID]=np.mean(dX)

            Ysums[r,offID,lngID]=Yp
            dYmeans[r,offID,lngID]=np.mean(dY)

            Zsums[r,offID,lngID]=Zp
            dZmeans[r,offID,lngID]=np.mean(dZ)


            if r==0 and lngID==20 and offID==35:
                plt.figure(figsize=(16,12))
                plt.subplot(2,1,1)
                plt.plot(X)
                plt.plot(Y)
                plt.plot(Z)
                plt.vlines([pSampleFrom, pSampleTo], -250, 1000)
                plt.ylim(-250,1000)
                plt.title("VCG")

                plt.subplot(2,1,2)
                plt.plot(dX)
                plt.plot(dY)
                plt.plot(dZ)
                plt.vlines([pSampleFrom, pSampleTo],-20, 20)
                plt.ylim(-20,20)
                plt.title("diff(VCG)")
                #
                # plt.subplot(3,1,3)
                # plt.plot(madc,'c-')
                # plt.plot(madcSmooth,'k-')
                #
                # plt.hlines(medLeft,0,l2)
                # plt.hlines(medRight,l2,len(madc))
                # plt.vlines([onset,offset],0,max(madc))

                plt.suptitle("Default offset and duratin for pExp (-0.275, 0.2)")
                plt.savefig("results/Default_pExp_range.png")
                plt.show()

                # figname = "figs/Dur_"+str(r)+"_"+data.File.values[r]+".svg"
                # plt.suptitle("Duration:%.3f"%duration+"; Area:%.3f"%area)
                # plt.savefig(figname)
                # plt.close()

    print("Done")
#
# data["Feas_QRSd"]=durations
# data["Feas_Area"]=areas
# data["Feas_expPADsum"]=expAdSums
# data["Feas_expPAreas"]=expPAreas

#data.to_pickle("Detected_113_cases_XYZQRS_durations.pkl")

basename = "offset_"+str(offsetMin)+"_"+str(offsetMax)+"_"+str(offsetSteps)+"_lng_"+str(lngMin)+"_"+str(lngMax)+"_"+str(lngSteps)
basename = basename.replace(".","")

expPAreaName = "results_data/COG_expPAreas_"+basename+".npy"
expPADSumName = "results_data/COG_expPADsums_"+basename+".npy"

expPADXName = "results_data/COG_expPADXs_"+basename+".npy"
expPADYName = "results_data/COG_expPADYs_"+basename+".npy"
expPADZName = "results_data/COG_expPADZs_"+basename+".npy"


xSumsName = "results_data/COG_xSums_"+basename+".npy"
dXmeansName = "results_data/COG_dXmeans_"+basename+".npy"
ySumsName = "results_data/COG_ySums_"+basename+".npy"
dYmeansName = "results_data/COG_dYmeans_"+basename+".npy"
zSumsName = "results_data/COG_zSums_"+basename+".npy"
dZmeansName = "results_data/COG_dZmeans_"+basename+".npy"

xAmpName = "results_data/COG_xAmps_"+basename+".npy"
yAmpName = "results_data/COG_yAmps_"+basename+".npy"
zAmpName = "results_data/COG_zAmps_"+basename+".npy"

Ysb0Name = "results_data/COG_ysb0_"+basename+".npy"

np.save(expPAreaName,expPAreas)
print("Done saving to ",expPAreaName)

np.save(expPADSumName,expPADsums)
print("Done saving to ",expPADSumName)

np.save(xSumsName,Xsums)
print("Done saving to ",xSumsName)
np.save(dXmeansName,dXmeans)
print("Done saving to ",dXmeansName)

np.save(ySumsName,Ysums)
print("Done saving to ",ySumsName)
np.save(dYmeansName,dYmeans)
print("Done saving to ",dYmeansName)

np.save(zSumsName,Zsums)
print("Done saving to ",zSumsName)
np.save(dZmeansName,dZmeans)
print("Done saving to ",dZmeansName)

np.save(xAmpName,Xamps)
np.save(yAmpName,Yamps)
np.save(zAmpName,Zamps)
print("Done saving amplitudes")

print("Saving expPADs...")
np.save(expPADXName,expPADXs)
np.save(expPADYName,expPADYs)
np.save(expPADZName,expPADZs)

print("Saving Ybs0s...")
np.save(Ysb0Name,Ysb0s)

print("Saved all")
