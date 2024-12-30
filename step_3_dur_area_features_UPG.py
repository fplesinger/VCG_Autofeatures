#Basic feature extractor for VCG Features project
#Extracts basic features as QRS area in separate components etc.
#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_proc


print("Python version: ")
import sys
print(sys.version)


pth = "results_data/COG_Detected_170_cases-all_XYZQRS.pkl"

print("Loading...")

data = pd.read_pickle(pth)

print("File loaded",data.columns)

rows = data.shape[0]

durations=[]
areas=[]
areasX=[]
areasY=[]
areasZ=[]
expAdSums=[]

expPADXs=[]
expPADYs=[]
expPADZs=[]

expPAreas=[]

xOffZero=[]
vvatOff=[]

negYsums=[]
negYSQsums=[]
negYmins=[]

for r in range(0,rows):
    print("Row",r)
    #if r<105:continue

    ecgDF = data.ECG.values[r]
    cn = np.array(ecgDF.columns[1:]).tolist()
    ecg = ecgDF.iloc[:,1:]
    #td = ecgDF.Time_sec.values[1]-ecgDF.Time_sec.values[0]
    #fs = 1/td
    fs=1000


    sr = int(fs*0.2)

    pExpStartOffset = int(fs*0.275)
    pExpLng = int(fs*0.2)

    xyz = data.XYZQRSshape.values[r]

    X = xyz[0,:]
    Y = xyz[1,:]
    Z = xyz[2,:]

    lng = len(X)

    negOrY = Y[int(lng/2)-100:int(lng/2)+100]
    negY = negOrY[negOrY < 0]
    negYsum = sum(negY)
    negYSQsum = sum(negY*negY)
    negYmin = min(negOrY)



    negYsums.append(negYsum)
    negYmins.append(negYmin)
    negYSQsums.append(negYSQsum)

    samplesFrom = int(lng/2-sr)
    sampleTo = int(lng/2+sr)

    pSampleFrom = int(lng/2-pExpStartOffset)
    pSampleTo = pSampleFrom+pExpLng


    Xp = X[pSampleFrom:pSampleTo]
    Yp = Y[pSampleFrom:pSampleTo]
    Zp = Z[pSampleFrom:pSampleTo]

    adXp = np.sum(np.abs(np.diff(Xp)))
    adYp = np.sum(np.abs(np.diff(Yp)))
    adZp = np.sum(np.abs(np.diff(Zp)))

    expPADSumX = adXp*adXp
    expPADSumY = adYp*adYp
    expPADSumZ = adZp*adZp

    expPADXs.append(expPADSumX)
    expPADYs.append(expPADSumY)
    expPADZs.append(expPADSumZ)

    expPADSum=np.sqrt(expPADSumX+expPADSumY+expPADSumZ)
    expAdSums.append(expPADSum)



    Xp=np.sum(np.abs(Xp-Xp[0]))
    Yp=np.sum(np.abs(Yp-Yp[0]))
    Zp=np.sum(np.abs(Zp-Zp[0]))

    expPArea=np.sqrt(Xp*Xp+Yp*Yp+Zp*Zp)
    expPAreas.append(expPArea)

    X = X[samplesFrom:sampleTo]
    Y = Y[samplesFrom:sampleTo]
    Z = Z[samplesFrom:sampleTo]

    adX = np.abs(np.diff(X))
    adY = np.abs(np.diff(Y))
    adZ = np.abs(np.diff(Z))

    mm = np.zeros((3,len(adX)))

    mm[0,:]=adX
    mm[1,:]=adY
    mm[2,:]=adZ

    madc = np.max(mm,axis=0)

    madX = mm[0,:]
    madY = mm[1,:]

    l2 = int(len(madc)/2)
    madcSmooth = signal_proc.filterMedian(madc,8)

    madYSmooth = signal_proc.filterMedian(madY,8)
    madXSmooth = signal_proc.filterMedian(madX,8)

    medYleft = np.median(madY[:l2])
    medXright = np.median(madX[l2:])

    medLeft = np.median(madc[:l2])
    medRight = np.median(madc[l2:])

    onset=-1

    for x in range(l2,0,-1):
        onset = x
        if madcSmooth[x]<medLeft:
            break

    offset = -1
    for x in range(l2,len(madc)):
        offset = x
        if madcSmooth[x]<medRight:
            break

    duration = (offset-onset)/fs
    durations.append(duration)


    #minimum X doprava
    sigX = xyz[0, :]


    xOffZeroPos = -1
    lng2 = int(len(sigX)/2)
    lngSamples = int(fs * 0.010)

    strPost = int(lng2+fs*0.005)

    #negace, pokud je střed vzhůru nohama?
    radC = int(0.1*fs)
    mnx=min(sigX[lng2-radC:lng2+radC])
    mxx = max(sigX[lng2-radC:lng2+radC])

    if abs(mnx)>abs(mxx):
        sigX=-sigX

    mr = int(0.02*fs)
    diffMid = np.mean(np.diff(sigX[int(lng2-mr):int(lng2+mr)]))
    vmid = sigX[strPost]


    if diffMid>0: #opatření,pokud to startuje na vzestupné hraně
        while sigX[strPost]>=vmid and strPost<len(sigX)-1:
            strPost +=1


    for x in range(strPost,len(sigX)-1):
        valpre = sigX[x-1] #np.mean(sigX[x-lngSamples:x])
        valpost = sigX[x] # np.mean(sigX[x:x+lngSamples])


        valpre = (sigX[x-2]+sigX[x-1])/2
        valpost = (sigX[x] + sigX[x + 1]) / 2

        xOffZeroPos = x

        if valpost>valpre:
            break

    postDur = (xOffZeroPos-lng2)/fs
    xOffZero.append(postDur)

    #nalezení toho samého před QRS:
    sigY = xyz[1,:]
    maxY = np.max(sigY)

    xPrePos=-1
    strs = int(lng2-fs*0.005)
    #lngSamples = int(fs * 0.010)

    diffRad = int(fs*0.02)
    peakSumDiff = np.sum(np.abs(np.diff(sigY[int(lng2-diffRad):int(lng2+diffRad)])))

    #diff1qrt = np.abs(np.mean(np.diff(sigY[:int(lng2/2)])))


    for xpos in range(strs,int(lng2/2),-1):
        xps = int(xpos)
        dataBlock = sigY[xps-lngSamples:xps]
        dataBlockPost = sigY[xps:xps+lngSamples]
        val = np.mean(dataBlock)
        valPost =np.mean(dataBlockPost)

        dVal = np.mean(np.diff(dataBlock))
        dValPost = np.mean(np.diff(dataBlockPost))

        #difposs = np.sum(np.abs(np.diff(sigY[int(xps-diffRad):int(xps+diffRad)])))

        if (maxY>0 and val>maxY/2): #or (maxY<0 and val<maxY/2): #původně jen v kladných číslech
            continue

        #if dVal < 0 and dValPost > 0:
        if dVal < dValPost/5 and dValPost > 0:   #688 komb
            #AUC 718 v kombinaci
        #if dVal<0 and dValPost>0 or difposs<peakSumDiff/20: #10: 672 samotné; 688 v kombinaci; #20: 699 v kombinaci
        #if (np.abs(np.mean(np.diff(dataBlock))))<diff1qrt:

            xPrePos=xps
            break


    predDur = (lng2-xPrePos)/fs




    if 1==0 and r<200:
        plt.figure()
        #plt.plot(sigY)
        plt.subplot(2,1,1)
        plt.plot(sigY,"k-")
        plt.vlines(xPrePos, -100, 100, colors="k")
        plt.vlines(1200,0,600,colors="cyan")
        if (not np.isnan(max(sigY))) and (not np.isnan(max(sigX))):
            plt.ylim(min(sigY),max(sigY))
        plt.ylabel("Y")

        plt.subplot(2,1,2)
        plt.plot(sigX,"b-")
        plt.vlines(xOffZeroPos, -100, 100,colors="b")
        plt.vlines(1200, 0, 600, colors="cyan")
        plt.ylabel("X")
        if (not np.isnan(max(sigY))) and (not np.isnan(max(sigX))):
            plt.ylim(min(sigX), max(sigX))

        #rh = data.Rhythm.values[r]
        plt.suptitle("Row:"+str(r)+data.File.values[r])
        plt.savefig("figs/base_row_"+str(r)+".png")

        plt.show()
        plt.close()



    vvatOff.append(predDur+postDur)

    #výpočet QRS area
    #baseline
    X=X-X[0]
    Y=Y-Y[0]
    Z=Z-Z[0]

    #všechny plochy do plusu, i záporné
    xabs=np.abs(X)
    yabs=np.abs(Y)
    zabs=np.abs(Z)

    #sumace
    sx = np.sum(xabs)/fs
    sy = np.sum(yabs)/fs
    sz = np.sum(zabs)/fs


    areaX = sx*sx
    areaY = sy*sy
    areaZ = sz*sz

    area = np.sqrt(areaX+areaY+areaZ)

    #carefull - area components are SQR!!!!! So only the total Area is valid as a number

    areas.append(area)
    areasX.append(areaX)
    areasY.append(areaY)
    areasZ.append(areaZ)


    if 1==0:

        plt.figure(figsize=(16,12))
        plt.subplot(3,1,1)
        plt.plot(X)
        plt.plot(Y)
        plt.plot(Z)
        plt.vlines([onset, offset], -1500, 1500)

        mx=1500
        mn=-1500

        if max(X)<100 and max(Y)<100 and max(Z)<100:
            mx=2
            mn=-2

        plt.ylim(mn,mx)

        plt.subplot(3,1,2)
        plt.plot(adX)
        plt.plot(adY)
        plt.plot(adZ)
        plt.vlines([onset, offset], 0, max(madc))

        plt.subplot(3,1,3)
        plt.plot(madc,'c-')
        plt.plot(madcSmooth,'k-')

        plt.hlines(medLeft,0,l2)
        plt.hlines(medRight,l2,len(madc))
        plt.vlines([onset,offset],0,max(madc))
        plt.ylim(mn,mx)
        #plt.show()

        figname = "figs/Dur_"+str(r)+"_"+data.File.values[r]+".svg"

        plt.suptitle("Duration:%.3f"%duration+"; Area:%.3f"%area)
        plt.savefig(figname)
        plt.close()

        print("Done")

data["Feas_QRSd"]=durations
data["Feas_Area"]=areas
data["Feas_AreaX"]=areasX
data["Feas_AreaY"]=areasY
data["Feas_AreaZ"]=areasZ
data["Feas_expPADsum"]=expAdSums

data["Feas_expPADx"]=expPADXs
data["Feas_expPADy"]=expPADYs
data["Feas_expPADz"]=expPADZs

data["Feas_expPAreas"]=expPAreas
data["Feas_vvatOff"]=vvatOff
data["Feas_xOffZero"]=xOffZero

data["Feas_negYsum"]=negYsums
data["Feas_negYSQsum"]=negYSQsums
data["Feas_negYmin"]=negYmins

newname = pth.replace(".pkl","_XO.pkl")

data.to_pickle(newname)
print("Done")

