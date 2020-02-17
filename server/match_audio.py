import pickle
from multiprocessing import Pool
from operator import itemgetter
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)
from tqdm.auto import tqdm

IDX_FREQ_I = 0
IDX_TIME_J = 1

######################################################################
# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

######################################################################
# Size of the FFT window, affects frequency granularity
DEFAULT_WINDOW_SIZE = 4096

######################################################################
# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
DEFAULT_OVERLAP_RATIO = 0.5

######################################################################
# Degree to which a fingerprint can be paired with its neighbors --
# higher will cause more fingerprints, but potentially better accuracy.
DEFAULT_FAN_VALUE = 15

######################################################################
# Minimum amplitude in spectrogram in order to be considered a peak.
# This can be raised to reduce number of fingerprints, but can negatively
# affect accuracy.
DEFAULT_AMP_MIN = 10

######################################################################
# Number of cells around an amplitude peak in the spectrogram in order
# for Dejavu to consider it a spectral peak. Higher values mean less
# fingerprints and faster matching, but can potentially affect accuracy.
PEAK_NEIGHBORHOOD_SIZE = 20

######################################################################
# Thresholds on how close or far fingerprints can be in time in order
# to be paired as a fingerprint. If your max is too low, higher values of
# DEFAULT_FAN_VALUE may not perform as expected.
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

######################################################################
# If True, will sort peaks temporally for fingerprinting;
# not sorting will cut down number of fingerprints, but potentially
# affect performance.
PEAK_SORT = True

######################################################################
# Number of bits to grab from the front of the SHA1 hash in the
# fingerprint calculation. The more you grab, the more memory storage,
# with potentially lesser collisions of matches.
FINGERPRINT_REDUCTION = 20


def spectrogram(data, plot=False):
    specgram = mlab.specgram(data,
                             NFFT=DEFAULT_WINDOW_SIZE,
                             Fs=DEFAULT_FS,
                             window=mlab.window_hanning,
                             noverlap=int(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO))[0]
    specgram = 10 * np.log10(specgram)
    specgram = specgram[20:1200, :]
    specgram[specgram == -np.inf] = 0
    if plot:
        plt.imshow(specgram.T, vmin=-100, vmax=100)
    else:
        return specgram


def get_2D_peaks(arr2D, plot=False, amp_min=DEFAULT_AMP_MIN):
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html#scipy.ndimage.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks (Fixed deprecated boolean operator by changing '-' to '^')
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = filter(lambda x: x[2] > amp_min, peaks)  # freq, time, amp
    # get indices for frequency and time
    frequency_idx = []
    time_idx = []
    for x in peaks_filtered:
        frequency_idx.append(x[1])
        time_idx.append(x[0])

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D.T)
        ax.scatter(frequency_idx, time_idx)
        ax.set_ylabel('Time')
        ax.set_xlabel('Frequency')
        ax.set_xlim(0, 500 - 50)
        ax.set_title("Spectrogram")
        ax.invert_yaxis()
        fig.set_size_inches(10, 10)
        plt.show()

    peaks = list(zip(frequency_idx, time_idx))
    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))
    return peaks


rate = 44100
amp_min = 20


def get_fragment(data, start, stop):
    return data[int(start * rate):int(stop * rate)]


# stations = [
#     wav.read('pod1.wav')[1],
#     wav.read('pod2.wav')[1],
#     wav.read('pod3.wav')[1],
# ]


def calculate_cost_3(reference_peaks, query_peaks_offset, match_threshold=(10, 20)):
    max_freq_diff, max_time_diff = match_threshold
    total = 0
    q_mask = np.ones(len(query_peaks_offset), dtype=bool)
    double_break = False
    for ref_peak in reference_peaks:
        #         print(ref_peak)
        for i_q, q_peak in enumerate(query_peaks_offset):
            if q_mask[i_q] == False or abs(ref_peak[1] - q_peak[1]) > max_time_diff:
                continue
            #             print(i_q)
            if abs(ref_peak[0] - q_peak[0]) < max_freq_diff:
                total += 1
                #                 print('yo')
                q_mask[i_q] = False
                double_break = True
                break
        if double_break:
            continue
    return total


def match_one(params):
    reference_peaks, query_peaks, offset, match_threshold = params
    query_peaks_offset = np.copy(query_peaks)
    query_peaks_offset[:, 1] = query_peaks_offset[:, 1] + offset
    cost = calculate_cost_3(reference_peaks, query_peaks_offset, match_threshold=match_threshold)
    return cost


def match_offset(reference_peaks, query_peaks, offset_max=25, offset_step=3, match_threshold=(10, 20)):
    query_peaks = np.array(query_peaks)
    reference_peaks = np.array(reference_peaks)
    offsets = range(-offset_max, offset_max, offset_step)
    params_iterable = ((reference_peaks, query_peaks, offset, match_threshold) for offset in offsets)
    costs = pool.map(match_one, params_iterable)
    return max(costs)


def peaks_offset_match_distance_precomputed(reference_peaks, query_peaks, offset_max=30, offset_step=3,
                                            match_threshold=(10, 20)):
    return match_offset(reference_peaks,
                        query_peaks,
                        offset_max=offset_max,
                        offset_step=offset_step, match_threshold=match_threshold)


#
# stations_short = [
#     stat[:(60 * rate)]
#     for stat in stations
# ]
#
# station_frags = [
#     [get_fragment(s, start, start + 5) for start in range(0, int(len(s) / rate - 5), 1)] for s in stations_short
# ]
#
# station_frags_starts = [
#     [start for start in range(0, int(len(s) / rate - 5))] for s in stations
# ]
#
#
# station_peaks = [
#     [get_2D_peaks(spectrogram(frag), plot=False, amp_min=amp_min)
#      for frag in tqdm(stat_frags)] for stat_frags in station_frags
# ]
# def match_station(stations, query, descriptor):

with open('peaks.3.short.pkl', 'rb') as f:
    station_peaks = pickle.load(f)

pool = Pool(10)


def match(query):
    query_peaks = get_2D_peaks(spectrogram(query), plot=False, amp_min=amp_min)
    station_match = [
        [peaks_offset_match_distance_precomputed(ref_peaks,
                                                 query_peaks,
                                                 offset_max=24,
                                                 offset_step=2,
                                                 match_threshold=(5, 5))
         for ref_peaks in tqdm(s)] for s in station_peaks
    ]


    station_best_match = [np.argmax(m) for m in station_match]
    for i, (sm, idx) in enumerate(zip(station_match, station_best_match)):
        # print(idx, len(station_match
        station_best_match[i] = sm[idx]
        if i > 0:
            station_best_match[i] += sm[idx-1]
        else:
            station_best_match[i] += sm[idx+2]

        if i < len(station_match) - 1:
            station_best_match[i] += sm[idx+1]
        else:
            station_best_match[i] += sm[idx-2]

    best_station = np.argmax(station_best_match)
    best_frag_idx = np.argmax(station_match[best_station])
    return best_station, best_frag_idx


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# def normalize()

if __name__ == '__main__':
    query = wav.read('pod2-1.normalized.wav')[1]


    def random_fragment(data, length=5):
        max_start = int(len(data) / rate) - length - 5

        true_start = np.random.randint(0, max_start)
        true_stop = true_start + length
        data_true = data[(true_start * rate):(true_stop * rate)]
        return data_true, true_start, true_stop


    query, qstart, qend = random_fragment(query, 5)
    print(qstart)
    print(match(query))
