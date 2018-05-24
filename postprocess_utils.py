import numpy as np
import librosa
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionErrorRate, DetectionAccuracy, DetectionPrecision, DetectionRecall
from pyannote.metrics.segmentation import SegmentationPurity, SegmentationCoverage


def deoverlap_predictions(predictions, features, hop_length):
    deoverlapped = [[] for i in range(len(features))]

    for i, f in enumerate(predictions):
        for j, p in enumerate(f):
            idx = (i * hop_length) + j
            if p >= 0.5:
                deoverlapped[idx].append(1)
            else:
                deoverlapped[idx].append(0)

    averaged = np.zeros(len(features))

    for i, p in enumerate(np.array(deoverlapped)):
        if len(p):
            averaged[i] = np.max(p)
        else:
            averaged[i] = 0

    return averaged


def defragment_vad(predictions):
    defragmented = np.zeros(len(predictions))
    mode = 0
    ones = 0
    zeros = 0
    for i, p in enumerate(predictions):
        defragmented[i] = mode
        if p == 0:
            ones = 0
            zeros += 1
            if zeros >= 50:
                mode = 0
                start = np.max((i - zeros, 0))
                defragmented[start:i] = [0 for _ in range(i - start)]
        else:
            ones += 1
            zeros = 1
            if ones >= 20:
                mode = 1
                start = np.max((i - ones, 0))
                defragmented[start:i] = [1 for _ in range(i - start)]

    return defragmented


def vad_voice_segments(predictions, frame_times):
    segments = []
    mode = 0
    start = 0
    for i, p in enumerate(predictions):
        if mode == 0:
            if p == 1:
                mode = 1
                start = frame_times[i]
        else:
            if p == 0:
                mode = 0
                segments.append((start, frame_times[i]))
    return segments


def vad_metrics(predictions,
                reference_segments,
                sr=22050,
                window_length=int(np.floor(0.032 * 22050)),
                hop_length=int(np.floor(0.016 * 22050))):
    frame_times = librosa.frames_to_time(range(len(predictions)), sr=sr, hop_length=hop_length, n_fft=window_length)
    predicted_segments = vad_voice_segments(predictions, frame_times)

    hypothesis = Annotation()
    for seg in predicted_segments:
        hypothesis[Segment(seg[0], seg[1])] = 1

    reference = Annotation()
    for seg in reference_segments:
        reference[Segment(seg[0], seg[1])] = 1

    precision = DetectionPrecision()(reference, hypothesis)
    error = DetectionErrorRate()(reference, hypothesis)
    recall = DetectionRecall()(reference, hypothesis)
    accuracy = DetectionAccuracy()(reference, hypothesis)

    metrics = {"precision": precision,
               "error": error,
               "recall": recall,
               "accuracy": accuracy}

    print(metrics)

    return metrics
