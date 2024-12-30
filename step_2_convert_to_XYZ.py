#XYZ convertor for VCG Features project
#Converts AVG QRS complexes into XYZ using Korrs transform
#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io


print("Python version: ")
import sys
print(sys.version)

print("Reading ")

flnm = "results_data/COG_Detected_170_cases-all.pkl"
data = pd.read_pickle(flnm)
print("Shape:",data.shape)
print(data.columns)

#Extract average beats
avgs=data.avgBeat.values




pth = "exportAVGS/"

print("Exporting to "+pth)

for i in range(avgs.shape[0]):
    print("Exporting avg shape ",i)

    mtr = avgs[i]

    file_path = pth+'avgShape'+str(i)+'_'+data.File.values[i]+'.mat'
    scipy.io.savemat(file_path, {'avgQRS': mtr})

    print()

print("Exporting done")


#X=0.38*I-0.07*II-0.13*V1+0.05*V2-0.01*V3+0.14*V4+0.06*V5+0.54*V6
#Y=-0.07*I+0.93*II+0.06*V1-0.02*V2-0.05*V3+0.06*V4-0.17*V5+0.13*V6
#Z=0.11*I-0.23*II-0.43*V1-0.06*V2-0.14*V3-0.20*V4-0.11*V5+0.31*V6

def lead12toXYZ(leadI,leadII,leadV1,leadV2,leadV3,leadV4,leadV5,leadV6):
    #korrs:
    X = 0.38*leadI-0.07*leadII-0.13*leadV1+0.05*leadV2-0.01*leadV3+0.14*leadV4+0.06*leadV5+0.54*leadV6
    Y =-0.07*leadI+0.93*leadII+0.06*leadV1-0.02*leadV2-0.05*leadV3+0.06*leadV4-0.17*leadV5+0.13*leadV6
    Z = 0.11*leadI-0.23*leadII-0.43*leadV1-0.06*leadV2-0.14*leadV3-0.20*leadV4-0.11*leadV5+0.31*leadV6

    return X,Y,Z

xyzLeads = []

for ri in range(data.shape[0]):
    print("Row",ri,"(",data.shape[0],")")

    ecgDF = data.ECG.values[ri]

    #cn = np.array(ecgDF.columns[1:])
    cn = np.array(ecgDF.columns[1:]).tolist()
    ecg = ecgDF.iloc[:,1:]

    #td = ecgDF.Time_sec.values[1]-ecgDF.Time_sec.values[0]
    fs = 1000

    outcome = data.Recidive.values[ri]==1

    s = data.avgBeat.values[ri]

    X,Y,Z = lead12toXYZ(s[:,cn.index("I")],s[:,cn.index("II")],s[:,cn.index("V1")],s[:,cn.index("V2")],s[:,cn.index("V3")],s[:,cn.index("V4")],s[:,cn.index("V5")],s[:,cn.index("V6")])

    xyzMember = np.zeros((3,len(X)))
    xyzMember[0,:]=X
    xyzMember[1,:]=Y
    xyzMember[2,:]=Z

    xyzLeads.append(xyzMember)

    if 1==1:

        fig = plt.figure(figsize=(16,8))

        plt.suptitle("Case " + str(ri + 1) + "/" + str(data.shape[0]) + " : " + data['File'].values[ri])

        #fs = data.Fs.values[ri]

        if not outcome:
            fig.patch.set_facecolor('xkcd:gray')


        lnd = int(fs*5)

        plt.subplot(3,1,1)

        plt.plot(ecg["V1"].values[:lnd])

        #plt.plot(ecgSignal[1000:3000])

        tcks = np.arange(0, lnd, fs)
        lbs = []

        for t in tcks:
            lbs.append(str(int(t / fs)))

        plt.xticks(tcks, lbs)
        plt.xlabel("Time [s]")
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(ecg["V1"].values[-lnd:])
        plt.xticks(tcks, lbs)
        plt.xlabel("Time [s] - od konce")
        plt.grid()


        plt.subplot(3,1,3)
        plt.plot(X,'r-')
        plt.plot(Y,'g-')
        plt.plot(Z, 'k-')

        plt.grid()

        avgSamps = len(X)


        tcks = np.arange(0, avgSamps + 1, fs / 20)
        lbs = []
        for t in tcks:
            lbs.append(str((t - avgSamps / 2) / fs))

        plt.ylabel('Averaged QRS')
        plt.xticks(tcks, lbs)
        plt.xlabel("Time [s]")
        plt.ylim(-750, 750)

        #plt.show()
        plt.savefig("figs\\Detect_" + str(ri + 1) + "_" + data['File'].values[ri] + "_XYZ.png")
        plt.close()

print("Converted")

data["XYZQRSshape"]=xyzLeads


data.to_pickle(flnm.replace(".pkl","")+"_XYZQRS.pkl")
print("Saved")

