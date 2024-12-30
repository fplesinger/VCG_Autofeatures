#QRS detector/AVG builder for VCG Features project
#Detects QRS complexes and builds QRS averaged shape
#This code uses AI model, described in https://www.nature.com/articles/s41598-022-16517-4 by Ivora, A. et al.
#The AI model cannot be published here since it was licences exclusively for the MDT-Medical Data Transfer (CZ) company.
#However, the code below shows how this single-lead model was used in multi-lead ECG signals.

#Author - Filip Plesinger/Adam Ivora, ISI of the CAS, CZ
#2020-2021


from collections import namedtuple
from enum import Enum

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
from scipy.signal import resample
from scipy.special import softmax
from scipy.stats import zscore


class QRSType(Enum):
    N_QRS = 0
    QRS_S = 1
    QRS_V = 2
    QRS_A = 3


QRS_MODEL_FS = 100
QRS_TYPES = [item.name for item in QRSType]
QRSPeak = namedtuple('QRSPeak', 'index type confidence')


def get_segmentation_centers(segmentation, ignore_class=QRSType.N_QRS.value, min_width=1):
    last = 0
    centers = dict()
    group_length = 1

    for i in range(len(segmentation)):
        if (segmentation[i] != last) or (i == len(segmentation) - 1):
            if (group_length >= min_width) and (last != ignore_class):
                centers[i - ((group_length + 1) // 2)] = last
            group_length = 1
            last = segmentation[i]
        else:
            group_length += 1
    return centers


def remove_close_peaks(probs, peaks, min_dist=int(0.15 * QRS_MODEL_FS)):
    locs = list(peaks.keys())

    for i in range(1, len(locs)):
        last_offset = 1

        # very uncommon case - happens when last peak was removed in previous iteration
        while locs[i - last_offset] not in peaks and (i - last_offset) >= 0:
            last_offset += 1

        curr, last = locs[i], locs[i - last_offset]
        dist_to_last = curr - last

        if dist_to_last < min_dist:
            class_curr = peaks[curr]
            class_last = peaks[last]

            if probs[class_curr, curr] <= probs[class_last, last]:
                del peaks[curr]
            else:
                del peaks[last]
    return peaks


def get_qrs_confidence(class_probs, indices):
    return 1 - class_probs[QRSType.N_QRS.value, indices]


def postprocess_qrs(qrs_output, input_fs):
    class_probs = softmax(qrs_output, axis=1)
    outcomes = np.argmax(qrs_output, axis=1)

    all_peaks = []
    for i in range(len(qrs_output)):
        peaks = get_segmentation_centers(outcomes[i])
        peaks = remove_close_peaks(class_probs[i], peaks)
        indices = np.array(list(peaks.keys()))
        confidences = get_qrs_confidence(class_probs[i], indices)

        peaks_orig_fs = (indices * (input_fs / QRS_MODEL_FS)).astype(int)
        all_peaks.append(list(map(QRSPeak._make, zip(peaks_orig_fs, peaks.values(), confidences))))

    if len(all_peaks) == 1:
        return all_peaks[0]
    return all_peaks


class QRSDetector:
    def __init__(self, filename='qrs_model_multiclass.onnx'):
        self.sess = rt.InferenceSession(filename)
        self.input_name = self.sess.get_inputs()[0].name

    def detect(self, input, input_fs):
        """
        Detect QRS peaks and their types.

        :param input: signal or a NumPy array of signals
        :param input_fs: sampling frequency
        :return: NumPy array of shape (number_of_qrs, 3)  (index, type, confidence) where the type string is QRS_CLASSES[type]
        """
        if input.ndim == 1:
            input = np.expand_dims(input, axis=0)
        input = input.astype(np.float32)
        input = zscore(input, axis=-1)
        input = resample(input, int(
            input.shape[-1] * (QRS_MODEL_FS / input_fs)), axis=-1)
        input = np.expand_dims(input, axis=1)
        qrs_output = self.sess.run(None, {self.input_name: input})[0]

        peaks = postprocess_qrs(qrs_output, input_fs)
        return peaks, qrs_output

    def detect_peaks_only(self, input, input_fs):
        """
        Detect QRS peaks independently on their types.

        :param input: signal or a NumPy array of signals
        :param input_fs: sampling frequency
        :return: list of indices of QRS peaks
        """
        peaks = self.detect(input, input_fs)
        return list(map(lambda x: x[0], peaks))


def transform_peaks(peaks):
    result = []
    for qrs_type in QRSType:
        result.append(peaks.T[0, peaks.T[1] == qrs_type])
    return result


def plot_result(orig_input, peaks):
    fig, ax = plt.subplots(figsize=(20, 4))

    plt.plot(orig_input, zorder=1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    handles = [mpatches.Patch(color=colors[type], label=QRS_TYPES[type]) for type in range(1, 4)]
    plt.legend(handles=handles)

    peaks = np.array(peaks)

    indices, types, confidences = peaks[:, 0].astype(np.int), peaks[:, 1].astype(np.int), peaks[:, 2]
    plt.scatter(indices, orig_input[indices], c=[colors[x] for x in types], zorder=2)

    for i, annotation in enumerate(confidences):
        ax.annotate(f'{annotation:.2f}', (indices[i], orig_input[indices[i]]))

    plt.show()


if __name__ == '__main__':
    orig_input = np.load('input.npy').astype(np.float32)[0]
    input_fs = 200

    detector = QRSDetector('qrs_model_multiclass.onnx')
    peaks = detector.detect(orig_input, input_fs)
    print(peaks)

    plot_result(orig_input, peaks)
