#AVG shape builder for VCG Features project
#Detects QRS complexes and builds QRS averaged shape. Uses AI QRS detector build for
#an automated holter ECG project. The detector could be replaced with some other (and public).

#The file with AI Detector neural network is named qrs_model_multiclass.onnx and is expected to reside in the project root

#Author - Filip Plesinger, ISI of the CAS, CZ
#17.3.2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import qrs_detector
import signal_proc as sp
from tools import pline

print("Python version: ")
import sys
print(sys.version)
pline()

from tools import pline, readTXT12lead
dataFile = "results_data/converted_prospective_170_files.pkl"

print("Reading data from", dataFile)
data = pd.read_pickle(dataFile)

print("Done", data.columns)
print("Cases:",data.shape[0])

#export table for rhythm file:
if 1==1:
    dataExp = data.copy()
    del dataExp["ECG"]
    #dataExp.to_excel("results_data/dataexp.xls")

pline()


QRSarray = []
avgBeat = []
f_bpm = []
f_bpm_std = []


#qdt = qrs_detector.QRSDetector()

det = None

data = data.reset_index()

for ri in range(data.shape[0]):
    #if ri<87:continue

    print("Detecting QRS for case", ri + 1, "/", data.shape[0])
    ecgCont = data['ECG'].values[ri]

    if ecgCont.shape[1]==14:
        signals = ecgCont.iloc[1:,1:-1].values
    else:
        signals = ecgCont.iloc[1:, 1:].values

    signals = np.transpose(signals)

    timecol=ecgCont.columns[0]
    if (timecol!="samplenr"):
        print("DIFFERENT TIMECOL:",timecol,"#########")


    secDelay = ecgCont[timecol].values[1]-ecgCont[timecol].values[0]



    #fs = 1/secDelay
    fs=1000

    usedLead = 0

    # peaks = qdt.detect()

    try:
        peaks = sp.detectQRSmultilead(signals, fs, det)
    except:
        print("ERROR DETECTING PEAKS","*"*100)
        peaks=[]

    # tady by mělo být zarovnání peaků na derivace
    winSize = int(0.2 * fs)
    winOff = int(0.1 * fs)
    iters = 5
    for p in range(len(peaks)):

        origPos = peaks[p]
        formerPos = origPos

        poses = np.zeros(iters)

        for it in range(iters):
            ds = origPos
            if origPos - winOff > 0:
                ds = origPos - winOff

            leadV1 = signals[6, ds:ds + winSize]
            leadV3 = signals[8, ds:ds + winSize]
            leadV6 = signals[11, ds:ds + winSize]

            cV1 = sp.compute_COG(np.abs(np.diff(leadV1)))
            cV3 = sp.compute_COG(np.abs(np.diff(leadV3)))
            cV6 = sp.compute_COG(np.abs(np.diff(leadV6)))

            cMean = (cV1 + cV3 + cV6) / 3
            delta = int(cMean - winSize / 2)

            # pokud by se to nemělo pohnout, tak to bude oprostřed cMean
            if origPos + delta - winOff + winSize < signals.shape[1]:
                origPos = origPos + delta

            poses[it] = origPos

            if delta == 0:
                break

        # zde to je již posunuté do polohy uprostřed QRS
        peaks[p] = origPos

        if ri == 1 and p == 0 and 1 == 1:
            plt.figure(figsize=(12, 8))
            strs = origPos - 1000
            ends = origPos + 1000
            sv1 = signals[6, :4000]
            sv3 = signals[8, :4000]
            sv6 = signals[10, :4000]
            plt.plot(sv1)
            plt.plot(sv3)
            plt.plot(sv6)

            for it in range(iters):
                plt.vlines(poses[it], -1000, 1000 + it * 200)

            plt.grid()
            plt.show()

        # tdlt = origPos-formerPos
        # print("Peak ",p,"delta loc",tdlt)

    ecg = signals[usedLead, :]
    # ecg = sp.filterFFT_bandPass(ecg,fs,0.1,40,False,'hamming')
    # peaks, envM, envH, envL, ignored, thr, kesRisks, thrL = sp.detectQRSoneLead(ecg,fs)

    print("Found", len(peaks), "QRS peaks")

    QRSarray.append(peaks)

    avgRad = 0.6
    avgSamps = avgRad * fs * 2

    avgQRS = np.zeros(int(avgSamps))

    numUsedPeaks = 0

    # multilead
    numChannels = signals.shape[0]
    avgQRS = np.zeros((int(avgSamps), numChannels))

    numUsedPeaks = 0

    for ld in range(0, numChannels):
        ecg = signals[ld, :]
        # one lead AVG
        for i in peaks:
            ss = i - avgSamps / 2
            es = i + avgSamps / 2

            if ss < 0 or es > len(ecg):
                continue

            numUsedPeaks += 1

            avgQRS[:, ld] = avgQRS[:, ld] + ecg[int(ss):int(es)]

        avgQRS[:, ld] = avgQRS[:, ld] / numUsedPeaks

    avgBeat.append(avgQRS)

    if len(peaks)>1:
        rrs = np.diff(peaks) / fs
        bpm = 60 / np.mean(rrs)
        std_bpm = np.std(60 / rrs)
    else:
        rrs=[]
        bpm = np.nan
        std_bpm=np.nan


    print("BPM %.2f" % bpm + " +/- %.2f" % std_bpm)

    f_bpm.append(bpm)
    f_bpm_std.append(std_bpm)

    if 1 == 1:

        outc = data['Recidive'].values[ri]==1

        fig = plt.figure(figsize=(30, 14))

        if not outc:
            fig.patch.set_facecolor('xkcd:gray')

        plt.subplot(3, 1, 1)
        plt.plot(ecg)

        plt.plot(peaks, ecg[peaks], 'ms')

        plt.vlines(peaks, np.min(ecg), np.min(ecg) + np.min(ecg) / 4, colors='m')
        plt.ylabel('ECG [AD units]')
        #+data['Channels'].values[ri][usedLead]+ #bylo to dřív dole
        plt.title("Filtered ECG. Multilead beat detection: " + str(len(peaks)) + " QRS complexes. BPM %.1f" % bpm + " +/- %.3f" % std_bpm)

        tcks = np.arange(0, len(ecg), fs * 1)
        lbs = []

        for t in tcks:
            lbs.append(str(int(t / fs)))

        plt.xticks(tcks, lbs)

        plt.xlabel("Time [s]")

        plt.subplot(3, 1, 2)

        plt.plot(peaks[:-1], rrs, 'mo')
        plt.ylim(0.2, 1.6)
        plt.xticks(tcks, lbs)
        plt.grid()

        plt.ylabel('RR [s]')

        plt.suptitle("Case " + str(ri + 1) + "/" + str(data.shape[0]) + " : " + data['File'].values[ri])

        nms = ecgCont.columns[1:]

        wantedECGNames=["V1","V2","V5","V6","aVL","aVR"]

        plt.subplot(3, 1, 3)

        sgMinMax=[]

        for c in range(numChannels):
            sg = avgQRS[:, c]
            sg -= np.median(sg)
            if c>=len(nms):
                print("ERROR: nms=",nms)
            channelName = nms[c]
            if not channelName in wantedECGNames:
                continue

            plt.plot(sg, label = channelName)

            sgMinMax= np.append(sgMinMax,sg)

        plt.grid()

        tcks = np.arange(0, avgSamps + 1, fs / 50) #puvodne fs/20
        lbs = []
        for t in tcks:
            lbs.append(str((t - avgSamps / 2) / fs))

        plt.ylabel('Averaged QRS')
        plt.xticks(tcks, np.array(lbs))
        plt.xlabel("Time [s]")
        plt.legend()

        if not np.isnan(bpm):
            plt.ylim(np.min(sgMinMax), np.max(sgMinMax))
            plt.xlim(0,1400)


            aRR = 60 / bpm;
            aRRsmp = int(avgSamps / 2 - aRR * fs)
            aRRsmp2 = int(avgSamps / 2 + aRR * fs)

            #       if aRRsmp>0:
            #           plt.vlines(aRRsmp, min(avgQRS),max(avgQRS))
            #           plt.vlines(aRRsmp2, min(avgQRS), max(avgQRS))
            # plt.show()

        sfl = data['File'].values[ri]
        sfl = sfl.replace(" ","").replace(".txt","").replace(".scp","")
        cnm = "figs\\COG_Detect_" + str(ri + 1) + "_" + sfl+ ".png"

        #plt.show()
        plt.savefig(cnm)
        plt.close()

print("QRS detection done")

data['QRS'] = QRSarray
data['avgBeat'] = avgBeat
data['Feas_BPM'] = f_bpm
data['Feas_BPM_STD'] = f_bpm_std

newName = "results_data/COG_Detected_"
data.to_pickle(newName + str(data.shape[0]) + "_cases-all.pkl")

print("Pickle saved")

