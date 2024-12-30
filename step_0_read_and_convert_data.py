#Data conversions for VCG Features project
#Converts ECG signals from TXT/CSV/PKL form into one global dataset
#It also reads summary info with outcome from dedicated excell file

#Variable mainPTH leads to parent patient folder.
#This folder contains one subfolder per patient (example: 00_SURNAME, 01_SURNAME etc.)
#Each subfolder contains two CSV files with ECG, named for example:00_SURNAME_pred.csv and 00_SURNAME_po.csv
#The "_pred" mean before electric cardioversion (ECV), "_po" means after ECV.
#Inside these CVSs, EKG signals are stored.
#Expected strucutre is samples in rows, Channels (regular 12-lead) in columns:

#samplenr	I	II	III	aVR	aVL	aVF	V1	V2	V3	V4	V5	V6
#0	32.5	-42.5	-75	7.5	52.5	-60	-115	-190	270	0	35	-237.5
#1	20	-47.5	-67.5	15	42.5	-57.5	-117.5	-202.5	265	-5	32.5	-235
#2	12.5	-50	-62.5	20	37.5	-57.5	-117.5	-207.5	255	-2.5	35	-235
#3	7.5	-47.5	-55	22.5	30	-52.5	-115	-212.5	250	2.5	42.5	-230

#The parent patient folder is expected to contain the file outcomes.xls.
# This excel sheet contains columns for number, surname, recidive (in Czech language)


#Author - Filip Plesinger, ISI of the CAS, CZ
#17.3.2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir, abspath
from unidecode import unidecode

from tools import pline, readTXT12lead, readPKL12lead,readCSV12lead

pline()
print("Convert prospective data v2")
print("Python version: ")
import sys
print(sys.version)
pline()

pline()

#dirs=[]

ecgs=[]
pths=[]
flnms=[]
states=[]
outcomes=[]

skipped=0


mainPTH = "" #INCLUDE PATH TO PATIENTS PARENT DIR HERE



notes = pd.read_excel(mainPTH+"\\outcomes.xls")

dirs = [name for name in listdir(mainPTH) if isdir(join(mainPTH, name))]

toDrop=[] #original code defined 3 duplicities to drop, but due to the anonymization it is no longer relevant. Cells 00_, 00_, 68_

for pth in dirs:

    if pth in toDrop:
        pline()
        print("Skipping",pth)
        skipped+=1
        pline()
        continue

    pth = join(mainPTH, pth)
    pline("-")
    files = [f for f in listdir(pth) if isfile(join(pth, f))]
    print(len(files),"files were found in the path",pth)


    print("Begin converting...")
    for fl in files:
        fid = files.index(fl)
        print("File",fid,"/",len(files),"\t"+fl, end="")

        isValid=False

        if fl[-3:]=="txt":

            data = readTXT12lead(join(pth, fl))
            isValid = True

        if fl[-3:]=="csv":

            data = readCSV12lead(join(pth, fl))
            isValid = True
            #print("Data shape:",data.shape)


        try:
            if fl[-3:]=="pkl":
                data = readPKL12lead(join(pth, fl))
                isValid = True
        except:
            print("*"*100," ERROR")
            continue


        measured="Unknown"

        uniFL = unidecode(fl)

        if " po" in uniFL.replace("_"," ").replace("-"," ").lower():
            measured="After"
        if " pre" in uniFL.replace("_"," ").replace("-"," ").lower():
            measured = "Before"

        if measured == "Unknown":
            print("*"*100,uniFL)

        outcome = float("nan")


        nt = notes[notes["číslo"]==float(fl[:2])]

        ri = fl.rindex("_")

        li = fl.index("_")

        nm = fl[(li+1):ri]

        nm = nm.capitalize()

        nt = notes[notes["jméno"] == nm]



        nt = nt[nt["číslo"]==float(fl[:li])]


        #notes["jméno"] = notes["jméno"].apply(unidecode)

        if nt.shape[0]==1:
            rec = nt.recidiva.values[0]
            if rec=="ano":outcome=1
            if rec=="ne":outcome=0

        if np.isnan(outcome):
            pline("%-NAN-")

        if isValid:
            pths.append(pth)
            flnms.append(fl)
            ecgs.append(data)
            states.append(measured)
            outcomes.append(outcome)
            print(".... added...***")
        else:
            print(" Skipped")
            skipped+=1

pline()
print("Conversion done")
print(len(ecgs),"files has been converted")
print(skipped,"files has been skipped")

pline("-")
unique, counts = np.unique(states, return_counts=True)
print("States:")
print(dict(zip(unique, counts)))

pline()


onm =  "results_data/converted_prospective_"+str(len(ecgs))+"_files.pkl"
print("Saving to",onm)
convData = pd.DataFrame()
convData["Path"]=pths
convData["File"]=flnms
convData["ECG"]=ecgs
convData["Measured"]=states
convData["Recidive"]=outcomes
convData.to_pickle(onm)
print("Done")




