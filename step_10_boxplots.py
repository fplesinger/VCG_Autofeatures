#Create boxplots/ROCs for final windows in VCG Features project
#This script generate PNG+SVG boxplots and ROC graphs with working windows of the selected
#features dX/dY/dZ Mean
#It also generates correlation matrix and results table

#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tools import pline

from sklearn import metrics
from scipy.stats import mannwhitneyu

flRS = "results_data/CV5_autofeatures_Data_85_cases_SMOOTH_median10_resTable.pkl"
resPos = pd.read_pickle(flRS)
print("Final positions loaded from ", resPos)
print("Columns", resPos.columns)

print("Reading main dataset with outcomes")
dnm = "results_data/COG_Detected_170_cases-all_XYZQRS_XO_KDE.pkl"
data = pd.read_pickle(dnm)
data=data[data.Measured=="Before"]
print("Done (with reduction). ",data.columns)
print("Shape:",data.shape)

#export table for Sabri
dfs = data.File.values
recs = data.Recidive.values

pacnums=[]
pacnames=[]
recidives=[]

for nms in dfs:
    strs = nms.split("_")

  #  if strs[0]=="57" and strs[1]=="ANONYMIZEDSURNAME": Commented due to manual fix and anonymization
  #      continue

    ind = dfs.tolist().index(nms)

    pacnums.append(strs[0])
    pacnames.append(strs[1])
    recidives.append(recs[ind])

ds = pd.DataFrame()
ds["Num"]=pacnums
ds["Surname"]=pacnames
ds["Recidive"]=recidives
ds.to_csv("results_data/export_84_for_Sabri.csv")
print("Pacient info exported")



dataLengths = []
pline()
print("Finding data lengths")

for row in range(data.shape[0]):
    lng = data.ECG.values[row].shape[0]
    dataLengths.append(lng)

pline()

#add XY loop area
if 1==0:
    print("Adding XY loop measure")
    XYLoopAreas = []
    from shapely import Polygon
    for ri in range(data.shape[0]):

        print("Loop area for",ri)
        try:

            XYZshape = data.XYZQRSshape.values[ri]
            X = XYZshape[0,:]
            Y = XYZshape[1,:]
            X[-1] = X[0]
            Y[-1]=Y[0]

            pgn = Polygon(zip(X,Y)) #X,Y



            XYLoopAreas.append(pgn.area)
        except:
            print("EXCEPTION")
            XYLoopAreas.append(0)

    data["XYLoopArea"]=XYLoopAreas

    nonresp = data.Recidive==True
    resp = data.Recidive==False

    dar = data[resp]
    dan = data[nonresp]

    valsN = dan.XYLoopArea.values
    valsR = dar.XYLoopArea.values

    t,p = mannwhitneyu(valsN,valsR)
    print("Area of XY loop:",p)





wantedFeas = ["dXMean","dYMean","dZMean"]


#scatter

plt.figure(figsize=(10,8))
sns.scatterplot(data=data, x="Feas_KDE_dZMean", y="Feas_KDE_dYMean", hue="Recidive")
plt.grid()
plt.title("Best KDE features by recidive")
plt.savefig("results/dYdZrecidive.png")
plt.show()


for fea in wantedFeas:
    pline()
    print("Analysing feature",fea)

    specRes = resPos[resPos.Name==fea]

    kdeOffsetRow = specRes["KDE offset row"].values[0]
    kdeDurationCol = specRes["KDE duration col"].values[0]

    print("Loaded ROW and COL:",kdeOffsetRow,kdeDurationCol)

    dnm = "results_data/COG_"+fea+"s_offset_-05_06_220_lng_001_05_98.npy"
    print("Loading data matrix")
    vals = np.load(dnm)
    print("Done. Shape:",vals.shape)
    vals = vals[:,kdeOffsetRow,kdeDurationCol]

    print("Differentiating by response")
    nonresp = data.Recidive==True
    resp = data.Recidive==False

    print("Adding feature to pickle")
    data["Feature_"+fea]=vals

    valsNRSP=vals[nonresp]
    valsRSP=vals[resp]

    valsRSP = valsRSP[~np.isnan(valsRSP)]
    valsNRSP = valsNRSP[~np.isnan(valsNRSP)]

    rspMean = np.mean(valsRSP)
    nrspMean = np.mean(valsNRSP)

    print("Mean responders:%.5f"%rspMean)
    print("Mean non-responders:%.5f"%nrspMean)

    wholeCol = np.hstack([valsRSP, valsNRSP]) * 1
    labels = np.hstack([np.ones(len(valsRSP)), np.zeros(len(valsNRSP))])
    try:
        fpr, tpr, thresholds = metrics.roc_curve(labels, wholeCol, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        stats, p = mannwhitneyu(valsRSP, valsNRSP)

        if auc<0.5:
            fpr, tpr, thresholds = metrics.roc_curve(labels, wholeCol, pos_label=0)
            auc = metrics.auc(fpr, tpr)




        print("AUC:%.2f"%auc," P=%.5f"%p)

        plt.figure(figsize=(10,8))
        plt.plot(fpr,tpr,linewidth=2)

        plt.fill_between(fpr, tpr, 0,alpha=0.2)
        plt.grid()
        plt.title(fea + "\nAUC:%.2f" % auc + " | Mann-Whit. U test => p:%.4f" % p)
        plt.plot([0,1],[0,1],"k:")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig("results/AUC_"+fea+".png")
        plt.savefig("results/AUC_" + fea + ".svg")
        plt.show()

    except:
        print("Error in inner OFFSET/LENGTH loop. Nans RSP:", sum(np.isnan(valsRSP)), " Nans NRSP:", sum(np.isnan(valsNRSP)))
        auc = 0.5
        p = 1

    plt.figure(figsize=(5,10))
    plt.boxplot([valsRSP,valsNRSP],widths=[0.4,0.4],showfliers=False)
    plt.title(fea+"\nAUC:%.2f"%auc+" | Mann-Whit. U test => p:%.4f"%p)
    plt.grid()

    plt.xticks([1, 2], ["Resp.\nN="+str(len(valsRSP)),"Non-r.\nN="+str(len(valsNRSP))])

    plt.savefig("results/resultFeature_boxplot_"+fea+".png")
    plt.savefig("results/resultFeature_boxplot_" + fea + ".svg")
    plt.show()

flnm = "results/Result_Table_with_Features_"+str(data.shape[0])+"_cases.pkl"
print("Saving resultant table as:",flnm)
data.to_pickle(flnm)

#data.to_excel(flnm.replace(".pkl",".xls"))
print("Done")

corrsData=data.iloc[:,-3:]
crr = corrsData.corr()

txt=["dXMean","dYMean","dZMean"]

plt.figure(figsize=(5,4))
plt.title("Correlation matrix")
cmap = cmap=sns.color_palette("Spectral", as_cmap=True)
sns.heatmap(crr,xticklabels=txt,yticklabels=txt,annot=True,vmin=-1,vmax=1,cmap=cmap)
plt.savefig("results/corrMatrix.png")
plt.show()

from scipy.stats import spearmanr, pearsonr

#add QRSd
corrsData["QRSd"] = data.Feas_QRSd.values


def correlation_matrix_with_significance(df):
    cols = df.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols)
    p_values = pd.DataFrame(index=cols, columns=cols)

    df.dropna(inplace=True)

    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                p_values.iloc[i, j] = 0.0
            else:
                corr, p_val = spearmanr(df.iloc[:,i].values, df.iloc[:,j].values)
                corr_matrix.iloc[i, j] = corr
                p_values.iloc[i, j] = p_val

    return corr_matrix, p_values


# Calculate correlation matrix and p-values
corr_matrix, p_values = correlation_matrix_with_significance(corrsData)

print("Correlation Matrix:")
print(corr_matrix)
print("\nP-Values Matrix:")
print(p_values)

print("="*80)

from scipy.stats import shapiro, kstest, normaltest


# Function to perform normality tests
def normality_tests(column):


    stat, p_shapiro = shapiro(corrsData[column])
    stat, p_kstest = kstest(corrsData[column], 'norm')
    stat, p_normaltest = normaltest(corrsData[column])

    print("-"*80)
    print(f'Normality tests for {column}:')
    print(f'Shapiro-Wilk Test p-value: {p_shapiro}')
    print(f'Kolmogorov-Smirnov Test p-value: {p_kstest}')
    print(f'D\'Agostino and Pearson\'s Test p-value: {p_normaltest}')


# Plot distributions and perform tests
for col in corrsData.columns:
    normality_tests(col)



