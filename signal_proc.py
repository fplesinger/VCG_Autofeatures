"""Library for Medisig standard processing
Author:  Filip Plesinger | ISI of the CAS,
2019, 2020
"""

import numpy as np
from scipy.signal import hilbert, windows
from scipy import signal
from scipy.signal import find_peaks
import qrs_detector as QRD



def cat(sig,fs):
    transSig = np.zeros(len(sig))

    ae = filterFFT_bandPass(sig, fs, 0, 20, True, 'tukey')

    # normalizace obálky, aby nezáleželo na amplitudě signálu
    if max(ae)==0:
        return transSig

    ae = ae / max(ae)



    hp = get_hypothetical_qrs(sig, fs, envM=ae)
    dl = len(sig)
    ws = int(0.1 * fs)

    method = 'pks'


    ns = len(hp)
    cm = np.zeros((ns, ns))
    aed = np.zeros(ns)

    for sx in np.arange(0, ns):

        cm[sx, sx] = 1

        x = int(hp[sx] - ws / 2)
        if x < 0:
            continue

        if x>dl-ws-1:
            continue

        # print(x, ' ', end='')

        dataX = sig[x:x + ws]
        vrX = max(dataX) - min(dataX)

        aea = max(ae[x:x + ws])
        aed[sx] = aea

        for sy in np.arange(0, sx):

            y = int(hp[sy] - ws / 2)
            if y < 0:
                continue

            if y > dl - ws - 1:
                continue

            dataY = sig[y:y + ws]

            vrY = max(dataY) - min(dataY)

            if len(dataX)!=len(dataY):
                print("sakryš")

            c = np.corrcoef(dataX, dataY)

            if vrY>0 and vrX>0:
                am = min([vrX / vrY, vrY / vrX])
            else:
                am=0

            cv = c[0, 1] * am

            cm[sx, sy] = cv
            cm[sy, sx] = cv



    # průměr:
    prm = np.mean(cm, axis=0)
    mlt = prm * aed
    mlt[mlt < 0] = 0
    transSig[hp] = mlt

    return transSig

def get_hypothetical_qrs(signal,fs,envM=[]):
    if len(envM)==0:
        envM = filterFFT_bandPass(signal, fs, 0, 22, True, 'tukey')

    minEnvM = np.percentile(envM, 20)
    win_secs = 0.15
    minDist = int(win_secs * fs)
    pks, amps = find_peaks(envM, height=minEnvM, distance=minDist)

    return pks


def recor_1D(signal, fs, peaks_to_test=[], window_size_sec=0.15, threshold=0.9):

    if len(peaks_to_test)==0:
        peaks_to_test = get_hypothetical_qrs(signal, fs)

    dl = len(signal)
    ws = int(window_size_sec * fs)
    sim1D = np.ones(dl)

    nthr, corrField = recor_atPoints(signal, peaks_to_test, ws, threshold)

    wr = int(round(float(ws) / 2))

    bw = windows.hamming(wr * 2)

    for i in range(len(peaks_to_test)):
        x = peaks_to_test[i]
        ss = x - wr
        se = x + wr

        if ss < 0 or se >= dl:
            continue

        sim1D[ss:se] += bw * nthr[i]


    return np.log(sim1D)

def recor_2D(signal, fs, peaks_to_test=[], thr=0.9, window_sizes_secs=[0.05, 0.1, 0.2, 0.5, 1, 2, 5]):
    #computes 2D simmilarity transformation (log)


    if len(peaks_to_test)==0:
        peaks_to_test = get_hypothetical_qrs(signal,fs)


    dl = len(signal)

    bigmap = np.ones((len(window_sizes_secs), dl))

    for wsec in window_sizes_secs:
        ws = int(wsec * fs)

        nthr, corrField = recor_atPoints(signal, peaks_to_test, ws, thr)

        #ntmap[wlist.index(wsec), :] = nthr

        wr = int(ws / 2)
        for i in range(len(peaks_to_test)):

            x = peaks_to_test[i]
            ss = x - wr
            se = x + wr

            add = windows.hamming(se - ss) * nthr[i]

            if se >= dl or ss < 0:
                continue
            bigmap[window_sizes_secs.index(wsec), ss:se] += add

    return np.log(bigmap)


def recor_atPoints(signal, pts, ws, thr=0.9):

    dl = len(signal)

    ns = len(pts)

    corrField = np.zeros((ns, ns))



    rad=int(round(ws/float(2)))

    for i in range(ns):

        s=pts[i]

        if (s-rad<0) or (s+rad)>=dl:
            continue

        # odebrat vzorový blok
        sbl = signal[s-rad:s + rad]

        #print("s=", s)



        corrField[i,i]=1

        # nechat ho prokorelovat přes celý signál
        for t in range(i):

            s2 = pts[t]

            if (s2-rad<0) or (s2+rad>=dl):
                continue

            tbl = signal[s2-rad:s2 +rad]



            corr = np.corrcoef(sbl, tbl)[0][1]

            if corr < 0:
                corr = 0

            corrField[i,t] = corr
            corrField[t,i] = corr


    # vyhledání počtu sloupců s korelací větší jak

    nthr = []


    for c in range(ns):
        column = corrField[:, c]
        nthr.append(len(column[column > thr]))

    return nthr,corrField


def recor(signal, fs, ws,sst, thr=0.9):
    dl = len(signal)
    ns = int((dl - ws) / sst) + 1

    corrField = np.zeros((ns, ns))

    si = 0
    ti = 0

    for s in range(0, dl - ws, sst):
        # odebrat vzorový blok
        sbl = signal[s:s + ws]

        print("s=", s)

        ti = 0

        # nechat ho prokorelovat přes celý signál
        for t in range(0, dl - ws, sst):
            tbl = signal[t:t + ws]

            corr = np.corrcoef(sbl, tbl)[0][1]

            if corr < 0:
                corr = 0

            corrField[ti, si] = corr

            ti += 1
        si += 1

    # vyhledání počtu sloupců s korelací větší jak

    nthr = []

    for i in range(int(ws / sst)):
        nthr.append(0)

    for c in range(ns):
        column = corrField[:, c]
        nthr.append(len(column[column > thr]))

    return nthr


def find_duration_to_zero(xStart,data):
    xl = xStart - 1
    sgnAct = np.sign(xStart)
    while xl >= 0 and np.sign(data[xl]) == sgnAct:
        xl -= 1

    xr = xStart + 1
    while xr < len(data) and np.sign(data[xr]) == sgnAct:
        xr += 1

    return xr-xl

def compute_COG(weights):
    sumW=0
    for x in range(len(weights)):
        sumW +=x*weights[x]

    sw = sum(weights)
    if sw==0:
        return 0

    return sumW/sw


def makePeriodic (ecg):
    """ Detrend ensuring that signal is periodic (needed for FFT): y[0]=y[end]
    """
    signalLength = len(ecg)-1
    y0 = ecg[0]
    yEnd = ecg[signalLength]
    totalDy = yEnd-y0

    dy = totalDy/signalLength

    for i in range(0,signalLength+1):
        sub = dy*i
        ecg[i] -= sub

    return ecg


def computeBPM (rrs):
    bpms=60/rrs

    meanBPM=np.mean(bpms)
    stdBPM=np.std(bpms)

    return meanBPM, stdBPM

def smooth3(signal):
    # tu obalku je potřeba vyhladit
    smE = np.zeros(len(signal))
    for k in range(1, len(smE) - 1):
        smE[k] = (signal[k - 1] + signal[k] + signal[k + 1]) / 3
    return smE

def filterMedian(signal,winSize):
    winRad=int(winSize/2)
    dataLen = len(signal)
    result=np.zeros(dataLen)

    result[:winRad] = signal[:winRad]
    result[dataLen-winRad:dataLen] = signal[dataLen-winRad:dataLen]

    for x in range(winRad,dataLen-winRad):
        result[x] = np.median(signal[x-winRad:x+winRad])

    return result

def filterMedianByDriver(signal,driverRadius):

    dataLen = len(signal)
    result=np.zeros(dataLen)

    for x in range(dataLen):
        winRad = int(driverRadius[x])
        st = x-winRad
        en = x+winRad

        if winRad<1:
            result[x] = signal[x]
        else:
            if st<0:st=0
            if en>=dataLen:en=dataLen-1
            result[x] = np.median(signal[st:en])
    return result


def detectQRSmultilead(ecg,fs,det):
    """ QRS Detection from multi-lead ECG data.
    Issues - single P-waves in AVB might be captured, noise is also partially captured
    """
    if det==None:
        det = QRD.QRSDetector()


    numLeads = ecg.shape[0]
    sumDets = np.zeros(ecg.shape[1])
    markRad = 0.05*fs
    markSamps = int(markRad)*2
    win =signal.hann(markSamps)

    for lead in range(numLeads):
        dt = ecg[lead,:]

        #peaks, envM, envH, envL, ignored, thr, kesRisks, thrL = detectQRSoneLead(dt, fs)
        retDetector = det.detect(dt,fs)
        peaks = retDetector[0]

        peaks = [p.index for p in peaks]


        for pk in peaks:
            ss = pk-markRad
            se = pk+markRad
            if ss<0 or se>=len(dt):
                continue
            sumDets[int(ss):int(se)] +=win

    mxp = max(sumDets)
    thr = round(mxp/2)

    peaks = signal.find_peaks(sumDets,height=thr)

    return peaks[0]


def detectQRSoneLead(ecg,fs,envH=[0]):
    """ QRS Detection from one-lead ECG data. Issues - single P-waves in AVB are captured, noise is also partially captured
    """

    env = filterFFT_bandPass(ecg, fs, 5, 25, True, 'boxcar')

    if len(envH)<2:
        if fs<=180:
            envH = filterFFT_bandPass(ecg, fs, 40, 48, True, 'boxcar')

        else:
            envH = filterFFT_bandPass(ecg, fs, 70, 90, True, 'hamming')


    env = env - envH * 3
    env[env<=0]=0

    envL = filterFFT_bandPass(ecg, fs, 1, 8, True, 'hamming')
    envM = env


    # oprava obálky proti ruchu

    env = np.clip(env, a_min=0, a_max=max(env))

    # detekce vrcholů, omezená vzájemno vzdáleností a thresholdem na 75. percentilu obálky
    dst = 0.2 * fs
    thr = np.percentile(env, 75)
    peaks, props = find_peaks(env, distance=dst, height=thr)
    thrL = np.percentile(envL, 75)

    # vyřazení vrcholů, které jsou určitě špatně

    dst = 0.45 * fs
    dst2 = 0.35 * fs
    tDel = []

    # pouze pro zobrazení
    xDel = []
    xDel2 = []
    xABC = []
    xSTD = []

    stdW = 0.05 * fs

    anyChange=True

    while anyChange:
        for p in range(0, len(peaks)):
            val = env[peaks[p]]
            pos = peaks[p]

            #if pos>2500:
            #    print("ere")

            #nepustit etremně úzké QRS - jsou to stimuly nebo ruch. Bacha, vyžaduje high/band pass!!!
            """"
            qrs_start=pos
            #doleva
            while qrs_start>0:
                qrs_start = qrs_start-1
                if np.sign(ecg[qrs_start]) != np.sign(ecg[pos]):
                    break
            qrs_end=pos
            #doprava
            while qrs_end<len(ecg):
                qrs_end = qrs_end +1
                if np.sign(ecg[qrs_end]) != np.sign(ecg[pos]):
                    break

            wdt = (qrs_end-qrs_start)/fs
            #print("X",pos,":W=",wdt)
            if wdt<0.08:
                tDel.append(p)
                xDel.append(peaks[p])
                continue

            """

            if pos < 0.5 * fs or pos > len(ecg) - 0.5 * fs:  # pokud je vrvchol blíž jak 0.5 sec ke kraji, tak mažu
                tDel.append(p)
                xDel.append(peaks[p])
                continue

            if p > 0:  # vrchol vlevo je příliš blízko s současně příliš silný => tento smažu
                distPre = peaks[p] - peaks[p - 1]
                if distPre < dst and env[peaks[p - 1]] > val * 6:
                    tDel.append(p)
                    xDel.append(peaks[p])
                    continue

            if p < len(peaks) - 1:  # vrhocl vpravo je blízko a současně příliš silný => tento mažu
                distPost = peaks[p + 1] - peaks[p]
                if distPost < dst and env[peaks[p + 1]] > val * 3:
                    tDel.append(p)
                    xDel.append(peaks[p])
                    continue

            if p > 0:  # vrchol vlevo je ještě blíž a současně ještě silější => tento smažu
                distPre = peaks[p] - peaks[p - 1]
                if distPre < dst2 and env[peaks[p - 1]] > val * 4:
                    tDel.append(p)
                    xDel.append(peaks[p])
                    continue

            if p < len(peaks) - 1:  # vrhocl vpravo je ještě blíž a současně ještě silnější => tento mažu
                distPost = peaks[p + 1] - peaks[p]
                if distPost < dst2 and env[peaks[p + 1]] > val * 2:
                    tDel.append(p)
                    xDel.append(peaks[p])
                    continue

            # analýza poměrů A(-0.15:-0.05); B(-0.05:0.05); C(0.05-0.15) #původně na surovém signálu....nemělo by být na obálce?
            a = int(pos - 3 * stdW)
            b = int(pos - stdW)
            c = int(pos + stdW)
            d = int(pos + 3 * stdW)
            blockA = ecg[a:b]
            blockB = ecg[b:c]
            blockC = ecg[c:d]

            blockEA = envM[a:b]
            blockEB = envM[b:c]
            blockEC = envM[c:d]
            blockEAD = envM[a:d]

            # rozdíl meanů vlevo a vpravo musí být menší, než půlka amplitudy celého kusu A:D. Toto odstraní NOISE - skoky v signálu
            meanA = np.mean(blockEA)
            meanC = np.mean(blockEC)
            ampABC = np.max(blockEAD) - np.min(blockEAD)

            if abs(meanA - meanC) > ampABC / 2:
                #tDel.append(p)
                #xDel.append(peaks[p])
                continue

            # odstranění P-vlny u AVB a FLUT:

            rangeA = np.max(blockA)-np.min(blockA)
            rangeB = np.max(blockB)-np.min(blockB)
            rangeC = np.max(blockC) - np.min(blockC)

            blockBel=envL[b:c]
            maxBel = np.max(blockBel)

            if rangeB<(rangeA+rangeC) and maxBel<thrL: #porovnání L obálky je potřeba, aby to nevyhazovalo komorovky
                tDel.append(p)
                xDel.append(peaks[p])
                continue


        peaks = np.delete(peaks, tDel)
        anyChange = len(tDel)>0
        tDel = []




    ignored = xDel

    # pro každý vrchol spočítám KESrisk v okně +/- 0.1 sec okolo tepu
    winRad=int(0.1*fs)

    kesRisks=np.zeros((len(peaks),1))

    for p in np.arange(0,len(peaks)):
        pos=peaks[p]
        a=pos-winRad
        b=pos+winRad
        sumE=sum(env[a:b])
        sumEL=sum(envL[a:b])
        kesRisks[p]=sumE/sumEL

        #print("I=%d; a=%d; b=%d; sumE=%.3f; sumEL=%.3f; ", p, a, b, sumE, sumEL);



    return peaks, envM, envH, envL, ignored, thr, kesRisks, thrL


def filterFFT_bandStop(sgnl,fs,fromHz,toHz):
    """"Simple bandStop FFT filter with rectangular window"""

    sp=np.fft.rfft(sgnl)

    N=len(sgnl);

    sh=N/fs;
    na=round(fromHz*sh);
    nb=round(toHz*sh);

    n2a=round(N-na);
    n2b=round(N-nb);

    sp[na:nb]=0;
    sp[n2b:n2a]=0;

    rek=np.fft.irfft(sp)
    return rek


def filterFFT_bandPass(sgnl,fs,fromHz,toHz, doEnvelope, winType):
    """FFT signal filter. sgnl: signal, fs: sampling frequency, fromHZ:toHz: frequency range, doEnvelope: set to True if hilbert transform should follow, winType: type of window to smooth frequency range
    winTypes:boxcar (=rectangular), traing, blackman, hamming, hann, nuttal ... + any else from scipy

    """
    sp=np.fft.rfft(sgnl)

    N=len(sgnl);

    sh=N/fs;

    fs2=round(N/2);

    na=round(fromHz*sh);
    nb=round(toHz*sh);

    #windowing function
    nw=nb-na
    if winType=='tukey':
        wind = signal.tukey(nw)
    else:
        wind=signal.get_window(winType,nw)

    sp[na:nb]=sp[na:nb]*wind


    n2a=round(N-na);
    n2b=round(N-nb);


    sp[0:na]=0;
    sp[nb:n2b]=0;
    sp[n2a:N-1]=0;

    rek=np.fft.irfft(sp)

    if doEnvelope:
        hlb=hilbert(rek)
        rek=np.sqrt(hlb.real*hlb.real+hlb.imag*hlb.imag)

    return rek