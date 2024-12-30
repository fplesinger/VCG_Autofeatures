#Create images with final windows in VCG Features project
#This script generate PNG+SVG images with working windows of the selected
#features dX/dY/dZ Mean
#It also generate some analysis not used in the manuscript as VCG loops for responders-non responders

#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn import metrics
from scipy.stats import mannwhitneyu

#by original step 17

#katalog tvarů VCG/ECG - VCG jsou jako stupně na kružnici
#http://cardiolatina.com/wp-content/uploads/2019/02/Class_8_Part_1_-_ECG_in_BBBs_-_LBBB_and_RBBB.pdf

def pline(chr="–",num=80):
    print(chr*num)


def prolongFrom1200To2400(osh):
    ns = np.zeros((3,2400))
    for i in range(1200):
        t0 = i*2
        t1 = i*2+1

        for d in range(3):
            ns[d,t0:t1+1]=osh[d,i]
    return ns

pline()

print("Loading data")
data = pd.read_pickle("results_data/COG_Detected_170_cases-all_XYZQRS.pkl")

data=data[data.Measured=="Before"]
data = data.drop(27) #Do not generate AVG shapes for this patient, signal is too damaged

print("Done. Columns N=",data.shape[1])
print(data.columns)



flRS = ""
if 1==1:
    flRS="results_data/CV5_autofeatures_Data_85_cases_SMOOTH_median10_resTable.pkl"
    resPos = pd.read_pickle(flRS)
    print("Final positions loaded from ",resPos)
    print("Columns",resPos.columns)


filters=[]

gt = "f"
for f in filters:
    gt=gt+f
gt=gt+"_"+str(data.shape[0])+"_cases"
gt=gt.replace("/","")

pline("=")
print("Analyzing...")

diffBy = "Recidive"

outcome = data[diffBy].values

RSP = outcome == False #Outcome je RECIDIVE, čili responder nemá recidivu
NRSP = outcome == True
print("Number responders:",sum(RSP))
print("Number of non-responders:",sum(NRSP))


pline("=")
print("Generating averaged shapes images")
pline()

QRS = data["XYZQRSshape"].values

pline("=")

shapeLng = QRS[0].shape[1]

avgResp = np.zeros((3,shapeLng))
avgNresp = np.zeros((3,shapeLng))

dr = data[RSP]
dn = data[NRSP]

numR = dr.shape[0]
#numR = numR-1 #hardcode!!! ,áe ve vnitřním poůi jednoho resp. s NAN. Resp. ID 18

for r in range(numR):
    valsR = dr["XYZQRSshape"].values[r]
    if sum(sum(np.isnan(valsR)))>0:
        print("HAS NANS!!!!")
    else:
        avgResp = avgResp+valsR/numR

numN = dn.shape[0]

numN -=1

for r in range(numN):
    valsN = dn["XYZQRSshape"].values[r]
    if sum(sum(np.isnan(valsN)))>0:
        print(r," HAS NANS!!!!")
    else:
        avgNresp = avgNresp+valsN/numN


dAR = np.diff(avgResp,axis=1)
dAN = np.diff(avgNresp,axis=1)

fs = 1000
xspc = fs/10

xtck = [x for x in range(0,1400,int(xspc))]
xlbs = ["%.1f"%((x/fs)-0.6) for x in xtck]


#samotný VCG QRS
if 1==0:

    id=0
    plt.figure(figsize=(12,4))
    xyz = data.XYZQRSshape.values[id]
    plt.plot(xyz[0,:],label="X")
    plt.plot(xyz[1,:],label="Y")
    plt.plot(xyz[2,:],label="Z")

    mn = np.min(xyz)
    mn = -200
    mx = 200

    plt.vlines(200,mn,mx,linestyles="-",colors="k")
    plt.vlines(600,mn,mx,linestyles="-",colors="k")
    plt.vlines(1200,-700,1700,linestyles=":",colors="k")

    #plt.xlim(0,1200)
    plt.ylim(-500,1000)


    plt.xticks(xtck,xlbs)
    plt.title("Averaged QRS complex (ID=%.0f"%id+")")
    plt.legend()
    plt.savefig("figs/AVG_example.svg")

    plt.show()


doSVG = True

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)

nrLS = ":"
if not doSVG: nrLS=":"

descResp = "(resp, N="+str(numR)+")"
descNon = "(non-resp, N="+str(numN)+")"

plt.plot(avgResp[0,:],"r-", label="X "+descResp)
plt.plot(avgNresp[0,:],"r"+nrLS,label ="X "+ descNon)

plt.xticks(xtck,labels=xlbs)

if not doSVG: plt.grid()
plt.legend()
plt.title("VCG:"+gt+" cases:"+str(data.shape[0])+" R:"+str(numR)+" N:"+str(numN))

plt.subplot(3,1,2)
plt.plot(avgResp[1,:],"g-", label="Y "+descResp)
plt.plot(avgNresp[1,:],"g"+nrLS,label = "Y "+descNon)
plt.xticks(xtck,labels=xlbs)
if not doSVG: plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(avgResp[2,:],"b-",label = "Z "+descResp)
plt.plot(avgNresp[2,:],"b"+nrLS,label = "Z "+descNon)
plt.xticks(xtck,labels=xlbs)
if not doSVG: plt.grid()
plt.legend()

if not doSVG:
    plt.savefig("figs/COG_Averaged_VCG_RN_"+gt+".png")
else:
    plt.savefig("figs/COG_Averaged_VCG_RN_" + gt + ".svg")

plt.show()



if 1==1:
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)

    plt.plot(dAR[0,:],"r-", label="X (resp)")
    plt.plot(dAN[0,:],"r:",label = "X (non-resp)")
    plt.xticks(xtck,labels=xlbs)

    plt.grid()
    plt.legend()
    plt.title("Diff VCG:"+gt+" cases:"+str(data.shape[0])+" R:"+str(numR)+" N:"+str(numN))

    plt.subplot(3,1,2)
    plt.plot(dAR[1,:],"g-", label="Y (resp)")
    plt.plot(dAN[1,:],"g:",label = "Y (non-resp)")
    plt.xticks(xtck,labels=xlbs)
    plt.grid()
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(dAR[2,:],"b-",label = "Z (resp)")
    plt.plot(dAN[2,:],"b:",label = "Z (non-resp")
    plt.xticks(xtck,labels=xlbs)
    plt.grid()
    plt.legend()

    plt.savefig("figs/COG_Diffed_averaged_VCG_RN_"+gt+".png")
    plt.show()

KDEFeatures = ["dXMean","dYMean","dZMean"]

starts=[]
ends=[]


for fnm in KDEFeatures:

    #fnm = "xSum"

    print("Feature",fnm)

    dzmr = np.where(resPos["Name"].values==fnm)[0][0]

    offset = resPos["KDE offset"].values[dzmr]
    duration = resPos["KDE duration"].values[dzmr]

    print("Offset %.3f"%offset," Duration %.3f"%duration)

    cntr=600
    offSample =cntr-offset*fs
    lng = duration*fs
    endSample = offSample+lng

    starts.append(offSample)
    ends.append(endSample)

    print("Start sample ",offSample," End sample ",endSample)
    pline()

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


if 1==0:
    plt.figure(figsize=(12,6))
    plt.title("detail dZ:"+gt+" cases:"+str(data.shape[0])+" R:"+str(numR)+" N:"+str(numN))
    plt.plot(dAR[2,:],"b-",label = "Z (resp)")
    plt.plot(dAN[2,:],"b:",label = "Z (non-resp")
    plt.xticks(xtck,labels=xlbs)

    plt.grid()
    plt.legend()
    plt.xlim(700,1700)
    plt.ylim(-3,3)
    plt.savefig("figs/detail_dZmean"+gt+".png")
    plt.show()

print("Done")

an = ["X","Y","Z"]
cl = ["r","g","b"]

#3D plot of loops
if 1==1:
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection="3d")
    ax.plot3D(avgResp[0, :], avgResp[1, :], avgResp[2, :], 'k-', label="Responders")
    ax.plot3D(avgNresp[0, :], avgNresp[1, :], avgNresp[2, :], 'k:', label="Non-responders")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


    fig = plt.figure(figsize=(14,14))
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)



    #shadow
    dummy = np.zeros(avgResp.shape[1])

    areaR = PolyArea(avgResp[0, :], avgResp[1, :])
    areaN = PolyArea(avgNresp[0, :], avgNresp[1, :])

    from shapely.geometry import Polygon
    pgon = Polygon(zip(avgResp[0, :], avgResp[1, :]))
    print("Area by shapely",pgon.area)

    print("Area by stackoverflow")
    print("AR:",areaR,"AN:",areaN)

    ax1.plot(avgResp[0, :], avgResp[1, :], 'k-', label="R-XY Area %.0f"%areaR)
    ax1.plot(avgNresp[0, :], avgNresp[1, :], 'k:', label="N-XY Area %.0f"%areaN)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid()




    ax3.plot(avgResp[1, :], avgResp[2, :], 'k-', label="R-YZ")
    ax3.plot(avgNresp[1, :], avgNresp[2, :], 'k:', label="N-YZ")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

    ax4.plot(avgResp[0, :], avgResp[2, :], 'k-', label="R-XZ")
    ax4.plot(avgNresp[0, :], avgNresp[2, :], 'k:', label="N-XZ")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Z")

    plt.legend()

    #plt.legend()
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 12))
plt.suptitle("Detail " + "sum" + gt + " cases:" + str(data.shape[0]) + " by "+diffBy+ " R:" + str(numR) + " N:" + str(numN))

ymins=[-40,-95,-40]
hghts=[90,280,80]

minsY=[-50,-100,-50]
maxsY=[300,200,100]

hatches=["/","\\","|","-","+","x","o","O",".","*"]



for axis in range(3):

    print("Printing subplot",axis)

    ax = plt.subplot(3,1,axis+1)
    tc = cl[axis]+"-"
    plt.plot(avgResp[axis,:],tc,label = an[axis]+"(" +diffBy+"== True)" )
    tc = cl[axis]+":"
    plt.plot(avgNresp[axis,:],tc,label =an [axis]+"(" +diffBy+"== False)" )
    plt.xticks(xtck,labels=xlbs)
    plt.ylabel(an[axis])

    #plt.vlines(offSample,-100,100,colors="k")
    #plt.vlines(endSample,-100,100,colors="k")

    h=0

    for ki in range(len(KDEFeatures)):


        kname = KDEFeatures[ki]
        ks = starts[ki]
        ke = ends[ki]

        print("Printing feature",kname )

        if an[axis].lower() in kname.lower():
            print("Should be drawn")
            clr = "gray"
            rect = patches.Rectangle((ks,ymins[axis]+h*50 ), ke-ks, hghts[axis], linewidth=2, edgecolor=clr, facecolor='none', label=kname)
            h+=1
            ax.add_patch(rect)

        #ax.add_patch(patches.Rectangle((100,100 ), 20, 100, linewidth=1, edgecolor=clr, facecolor='black'))

    #plt.grid()
    plt.legend()
    plt.grid()
    #plt.xlim(0,1200)

    plt.ylim(minsY[axis],maxsY[axis])
print("Done")

gnm="figs/detail_" + an[axis] + "sum" + gt
plt.savefig(gnm + ".png")
plt.savefig(gnm + ".svg")
plt.show()
