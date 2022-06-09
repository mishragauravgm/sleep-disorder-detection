#Code to generate data for with subject information, to do LOSO

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')



import scipy
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np
import matplotlib.pyplot as plt
## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a
## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a
def notch_filter(cutoff, q):
    nyq = 0.5*fs
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a

def highpass(data, fs, order=5):
    b,a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b,a,data)
    return x

def lowpass(data, fs, order =5):
    b,a = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(b,a,data)
    return y

def notch(data, powerline, q):
    b,a = notch_filter(powerline,q)
    z = lfilter(b,a,data)
    return z

def final_filter(data, fs, order=5):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order = order)
    y = lfilter(d, c, x)
    f, e = notch_filter(powerline, 30)
    z = lfilter(f, e, y)     
    return z

orignal_fs = 360

fs = 256
#ecg_signal = scipy.signal.resample(ecg_signal_data, num=fs*60*5)[:1280]
# ecg_signal.shape

## Order of five works well with ECG signals
cutoff_low = 20
cutoff_high = 0.5
powerline = 50
order = 5


def get_time_series_features(signal):
    window_size = len(signal)
    # mean
    sig_mean = np.mean(signal)
    # standard deviation
    sig_std = np.std(signal)
    # avg absolute difference
    sig_aad = np.mean(np.absolute(signal - np.mean(signal)))
    # min
    sig_min = np.min(signal)
    # max
    sig_max = np.max(signal)
    # max-min difference
    sig_maxmin_diff = sig_max - sig_min
    # median
    sig_median = np.median(signal)
    # median absolute deviation
    sig_mad = np.median(np.absolute(signal - np.median(signal)))
    # Inter-quartile range
    sig_IQR = np.percentile(signal, 75) - np.percentile(signal, 25)
    # negative count
    sig_neg_count = np.sum(s < 0 for s in signal)
    # positive count
    sig_pos_count = np.sum(s > 0 for s in signal)
    # values above mean
    sig_above_mean = np.sum(s > sig_mean for s in signal)
    # number of peaks
    sig_num_peaks = len(find_peaks(signal)[0])
    # skewness
    sig_skew = stats.skew(signal)
    # kurtosis
    sig_kurtosis = stats.kurtosis(signal)
    # energy
    sig_energy = np.sum(s ** 2 for s in signal) / window_size
    # signal area
    sig_sma = np.sum(signal) / window_size

    return [sig_mean, sig_std, sig_aad, sig_min, sig_max, sig_maxmin_diff, sig_median, sig_mad, sig_IQR, sig_neg_count, sig_pos_count, sig_above_mean, sig_num_peaks, sig_skew, sig_kurtosis, sig_energy, sig_sma]


def get_freq_domain_features(signal):
    all_fft_features = []
    window_size = len(signal)
    signal_fft = np.abs(np.fft.fft(signal))
    # Signal DC component
    sig_fft_dc = signal_fft[0]
    # aggregations over the fft signal
    fft_feats = get_time_series_features(signal_fft[1:int(window_size / 2) + 1])

    all_fft_features.append(sig_fft_dc)
    all_fft_features.extend(fft_feats)
    return all_fft_features

data_csv = []
files = os.listdir('/home/mishra.g/spring2022/hci/project/data/wake_ecg_split_const_time/')
count = 1;

print(f'Starting writing CSVs with subject info, processing {len(files)} files, stay tuned....\n')
    

for i in glob.glob('/home/mishra.g/spring2022/hci/project/data/wake_ecg_split_const_time/*.csv'):
    filename = os.path.basename(i);
    data = pd.read_csv(i, header=None)
    arr = list(final_filter(data.T.values[0,:],fs,order))
    
    features = []
    features.extend(get_time_series_features(arr))
    features.extend(get_freq_domain_features(arr))
    arr=[]
    if(abs(data.mean()[0])>50):
        continue;


    if(filename[1]=='r'):
        name = filename[0:5];
        arr.extend([name])
        arr.extend(features)
        arr.extend([1])
    elif(filename[1]=='n'):
        name = filename[0:4];
        arr.extend([name])
        arr.extend(features)
        arr.extend([2])
    elif(filename[1]=='a'):
        name = filename[0:6];
        arr.extend([name])
        arr.extend(features)
        arr.extend([3])
    elif(filename[1]=='f'):
        name = filename[0:6];
        arr.extend([name])
        arr.extend(features)
        arr.extend([4])
    elif(filename[1]=='l'):
        name = filename[0:5];
        arr.extend([name])
        arr.extend(features)
        arr.extend([5])
    elif(filename[1]=='b'):
        name = filename[0:5];
        arr.extend([name])
        arr.extend(features)
        arr.extend([6])
    elif(filename[1]=='d'):
        name = filename[0:4];
        arr.extend([name])
        arr.extend(features)
        arr.extend([7])
    else:
        name = filename[0:3];
        arr.extend([name])
        arr.extend(features)
        arr.extend([0]);
    data_csv.append(arr);
    count=count+1;
    if count%5000==0:
        print(f'Done {count}/{len(files)}, saving csv for safety, in case code fails!')
        df = pd.DataFrame(data_csv)
        df.to_csv('/home/mishra.g/spring2022/hci/project/all_data_with_subject_info.csv',index=False);
df = pd.DataFrame(data_csv)
df.to_csv('/home/mishra.g/spring2022/hci/project/all_data_with_subject_info.csv',index=False);