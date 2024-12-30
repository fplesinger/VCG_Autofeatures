#Computes statistics in VCG Features project
#This script computes AUC and Mann-Whithney U-test values for all features, both clinical and ECG/VCG-derived.
#It requires link to CSV with clinical features (clin_name) and PKL with computed parameters (compnam)
#Author - Filip Plesinger, ISI of the CAS, CZ
#2022-23


import numpy
import pandas as pd
from colorama import Fore,Back

clin_name = "data/EKV_SH_final_QRSdR_converted_added_missing.csv"

print("Reading clinical table from",clin_name)
dc = pd.read_csv(clin_name, sep=";")
print("Done. Columns:",dc.columns)





print("="*80)

compnam = "results/Result_Table_with_Features_85_cases.pkl"

print("Reading results from",compnam)
dr = pd.read_pickle(compnam)
print("Done. Columns:",dr.columns)





def insert_sep_line(file_path):
    # Read the original content of the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Insert the new line at the beginning
    content.insert(0, "SEP=,\n")

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(content)

def returnDRRawRowIndex(patname):
    names = dr.File.values

    occs=[]

    for rid in range(len(names)):
        nm = names[rid].lower().split("_")[1]



        if patname.lower()==nm:
            occs.append(rid)

    return occs


def returnDCRawRowIndex(filename):
    names = dc["Příjmení"].values

    nmDR = filename.lower().split("_")[1]

    #if nmDR=="ANONYMIZEDSURNAME_ANONYMIZEDNAME":  Commented due to manual fix and anonymization
    #    nmDR = "ANONYMIZEDSURNAME"


    occs=[]

    for rid in range(len(names)):
        nm = names[rid]
        if nmDR==nm:
            occs.append(rid)

    if len(occs)==0:
        occs.append(9999)

    return occs


print("Combining")
clinnames = dc["Příjmení"].values

indexesToDR=[]
for nm in clinnames:
    rri = returnDRRawRowIndex(nm)
    print("Raw row index for",nm,"is",rri)
    if len(rri)==1:
        indexesToDR.append(rri[0])
    else:
        print("Patient not found!!!")
        indexesToDR.append(999)
dc["drRowIndex"]=indexesToDR

import numpy as np
dfs = np.diff(indexesToDR)


print("="*80)

filenames = dr.File.values

indexesToDC=[]

QRSdRD = []
female_g=[]


clinData={}


for nm in filenames:
    rri = returnDCRawRowIndex(nm)

    print("Raw row index for",nm,"is",rri)

    indexesToDC.append(rri[0])
    if (rri[0]==9999):
        print("**********")
        QRSdRD.append(numpy.NaN)

    else:
        QRSdRD.append(dc.QRSd.values[rri[0]])


    #fill all clin data
    for c in dc.columns[6:]:
        if not c in clinData.keys():
            clinData[c]=[]

        if rri[0]<9999:
            clinData[c].append(dc[c].values[rri[0]])
        else:
            clinData[c].append(np.NaN)


for k in clinData.keys():
    dr[k]=clinData[k]


dr["IdxToDC"]=indexesToDC
dr["QRSdRadek"] = QRSdRD


#attach new data from Sabri:
nds ="data/export_84_for_Sabri_Recidives_FU6M.csv"
print("Loading Sabri last data from",nds)
dsb = pd.read_csv(nds,sep=";")
print("Done. ",dsb.columns)

dr["EF"]=dsb.EF.values
dr["VPTH"]=dsb.VPTH.values
dr["R6MFU"]=dsb["6MFU"].values

#dr[]

drs = dr.sort_values(by="IdxToDC",ascending=True)
rw = np.diff(drs["IdxToDC"].values)




txt=["dXMean","dYMean","dZMean"]



from scipy.stats import spearmanr, pearsonr, pointbiserialr

#add QRSd
corrsData = dr[["Feature_dXMean","Feature_dYMean","Feature_dZMean"]]

def correlation_matrix_with_significance(df, method="spearman"):
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
                if method=="spearman":
                    corr, p_val = spearmanr(df.iloc[:,i].values, df.iloc[:,j].values)
                if method == "biserial":
                    corr, p_val = pointbiserialr(df.iloc[:, i].values, df.iloc[:, j].values)
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


#dr["dcRowIndex"]=indexesToDR


dr = dr.drop(index=83)

dr["EHRA>1"]=(dr.ehra_an>1)*1
dr["EF<40"]=(dr.EF<0.4)*1

dr["VPTH_2"]=(dr.VPTH==2)*1
dr["VPTH_3"]=(dr.VPTH==3)*1

print("="*80)
#add QRSd
corrsData = dr[["Feature_dXMean","Feature_dYMean","Feature_dZMean","EF<40","VPTH_2","VPTH_3"]]
corr_matrix, p_values = correlation_matrix_with_significance(corrsData, method="biserial")

print("Correlation Matrix from point-biserial corr:")
print(corr_matrix)
print("\nP-Values Matrix:")
print(p_values)

print("="*80)

dr.loc[7,"female_G"]=1 #specific case fix
dr.loc[161,"female_G"]=0 #speific case fix

Rec36 = []
rowToDel=[]

for ri in range(dr.shape[0]):
    v6 = dr.R6MFU.values[ri]
    #v3 = dr.Recidive.values[ri]

    if isinstance(v6,float) and np.isnan(v6):
        rR6=np.nan
        rowToDel.append(ri)
    else:
        rR6 = ("AF" in v6)*1
    Rec36.append(rR6)

dr["Recidive36M"]=Rec36

if False:
    print("Exporting final combined dataset")
    expname = "VCG_Data_Final_Eval.pkl"
    dr.to_pickle(expname)
    dr.to_csv(expname.replace(".pkl",".csv"))
    print("Final table has been written to",expname)


print("Prepare final folder with ECG data")
import shutil
import os
for i in range (dr.shape[0]):
    print("Copying record #",i)

    epth = dr.Path.values[i]
    enm = dr.File.values[i]
    full_path_source = os.path.join(epth, enm)
    full_path_dest = os.path.join(r"M:\d03\filip\pokusy\ai_fs\software-prospective\data\exportedECG", enm)

    shutil.copy(full_path_source,full_path_dest)

print("Copying done")

trgt = "Recidive"

trgt = "Recidive36M"

if "36M" in trgt:
    dr=dr.drop(dr.index[rowToDel])

dr_RESP = dr[dr[trgt]==False]
dr_NRESP = dr[dr[trgt]==True]


print("="*80)

print("")

tb="\t"

fnm = []
mean_alls = []
std_alls=[]
sum_alls=[]
mis_alls=[]

mean_resp = []
std_resp=[]
sum_resp=[]

mean_nresp = []
std_nresp=[]
sum_nresp=[]

p_rn=[]
auc_rn=[]

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

for c in dr.columns[11:]:
    print("-"*80)
    print("Feature ",c)
    if c=="R6MFU":
        print("Skipping")
        continue

    fnm.append(c)



    va=dr[c].values

    if c=="Recidive36M":
        print()

    vac = va[~np.isnan(va)]
    print("N=",len(vac))

    isBin = len(np.unique(vac))==2


    if "ehra" in c:
        print()

    if isBin:
        sum_alls.append(np.nansum(va))
        mean_alls.append(np.NaN)
        std_alls.append(np.NaN)
    else:
        sum_alls.append(np.NaN)
        mean_alls.append(np.nanmean(va))
        std_alls.append(np.nanstd(va))

    numMissing = np.sum(np.isnan(va))
    mis_alls.append(numMissing)

    print("Missing:",numMissing)

    vr=dr_RESP[c].values



    print("Responders: N=",len(vr),end=" ")

    if isBin:
        mean_resp.append(np.NaN)
        std_resp.append(np.NaN)
        sum_resp.append(np.nansum(vr))

        print("BIN: sum=",np.nansum(vr), end=" ")

    else:
        mean_resp.append(np.nanmean(vr))
        std_resp.append(np.nanstd(vr))
        sum_resp.append(np.NaN)

        print("CONT: mean=%.3f"%np.nanmean(vr),"+/-%.3f"%np.nanstd(vr),end=" ")


    vn=dr_NRESP[c].values
    print(" | Non-responders: N=",len(vn), end=" ")

    if isBin:
        mean_nresp.append(np.nan)
        std_nresp.append(np.nan)
        sum_nresp.append(np.nansum(vn))
        print("BIN: sum=", np.nansum(vn), end=" ")
    else:
        mean_nresp.append(np.nanmean(vn))
        std_nresp.append(np.nanstd(vn))
        sum_nresp.append(np.nan)
        print("CONT: mean=%.3f"%np.nanmean(vn), "+/-%.3f"%np.nanstd(vn), end=" ")


    vrc = vr[~np.isnan(vr)]
    vnc = vn[~np.isnan(vn)]

    print()


    #mann-whithney-u test
    stat, p_value = mannwhitneyu(vrc, vnc)



    if (p_value<0.05):
        print(Back.CYAN,end="")
    print("MWU: %.5f"%p_value,Back.RESET)

    p_rn.append(p_value)

    sf = dr[["Recidive",c]]

    sf=sf.dropna()

    auc = roc_auc_score(sf.Recidive, sf.iloc[:,1])
    if auc<0.5:
        auc = 1-auc

    print(f'AUC: {auc}')

    auc_rn.append(auc)

    #print(c,mean_ALL)


dTab1 = pd.DataFrame()

dTab1["Feature"]=fnm

dTab1["All-Sum"] = sum_alls
dTab1["All-Mean"] = mean_alls
dTab1["All-Std"]=std_alls
dTab1["All-missing"]=mis_alls



dTab1["RESP-Sum"] = sum_resp
dTab1["RESP-Mean"] = mean_resp
dTab1["RESP-Std"]=std_resp
#dTab1["RESP-missing"]=mis_resp

dTab1["NRESP-Sum"] = sum_nresp
dTab1["NRESP-Mean"] = mean_nresp
dTab1["NRESP-Std"]=std_nresp
#dTab1["RESP-missing"]=mis_resp


dTab1["p_RESPxNRESP"]=p_rn
dTab1["auc_RESPxNRESP"]=auc_rn

flnm = "results/tab1_"+trgt+".csv"

dTab1.to_csv(flnm)
insert_sep_line(flnm)


print("File saved and SEP line added into",flnm)








