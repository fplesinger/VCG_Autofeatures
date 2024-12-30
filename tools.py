#Utilities for VCG Features project
#Author - Filip Plesinger, ISI of the CAS, CZ
#17.3.2023


import numpy as np
import pandas as pd
import sys
import argparse

def pline(chr="=",num=80):
    print(chr*num)


def readPKL12lead(flnm, defaultNames=["Time_sec","I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]):


    ecgDF = pd.read_pickle(flnm)
    return ecgDF

    #printer = ScpPrinter()

def readTXT12lead(flnm, defaultNames=["Time_sec","I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]):


    file = open(flnm, mode='r')
    tx = file.read()
    file.close()


    rows = tx.split("\n")
    cols=rows[0].split(",")

    dataRowCount = len(rows)-2

    ecg=np.zeros((dataRowCount,len(cols)))

    for ri in range(1,len(rows)-1):
        datalineStr = rows[ri].split(",")
        floatLine = np.asarray(datalineStr,float)
        ecg[ri-1,:]=floatLine


    ecgDF = pd.DataFrame(data=ecg, columns=defaultNames)

    return ecgDF


def readCSV12lead(flnm, defaultNames=["Time_sec","I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]):




    ecgDF = pd.read_csv(flnm,delimiter='\t')

    if (ecgDF.shape[1]==1):
        pline("#")
        print("Delimiter set to ;")
        ecgDF = pd.read_csv(flnm, delimiter=';')
        pline()


    return ecgDF