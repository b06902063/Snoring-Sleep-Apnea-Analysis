import librosa
import os
import numpy as np
import math
import pywt
from PyEMD import EMD

def read_folder(folder_path):
    in_folder = os.listdir(folder_path)
    filenames = [x for x in in_folder if x[-4:] == '.wav']
    return filenames
    
def read_wav(filename):
    x, sr = librosa.load(filename, sr = None, mono = True,  )
    return x,sr
    
    
    
def crest_factor(data, win_size):
    data_matrix = librosa.util.frame(data, win_size, win_size)
    v90 = np.percentile(np.absolute(data_matrix), 90, axis = 0)
    v10 = np.percentile(np.absolute(data_matrix), 10, axis = 0)
    data_matrix1 = data_matrix[data_matrix >= v10]
    data_matrix2 = data_matrix[data_matrix < v90]
    RMS = np.sqrt(np.mean(np.square(data_matrix2), axis = 0))
    return np.divide(v90, RMS)
    
def get_formants(data, winsize):
    data_matrix = librosa.util.frame(data, winsize, winsize, axis = 0)
    formants = []
    for frame in data_matrix:
        frame = np.asfortranarray(frame)
        A = librosa.core.lpc(frame,18)
        rts = np.roots(A)
        rts = rts[np.imag(rts)  >= 0]
        angz = np.arctan2(np.imag(rts),np.real(rts))
        frqs = angz * winsize / (2 * np.pi)
        frqs.sort()
        formant = frqs[:3]
        formants.append(formant)
    formants = np.array(formants)
    return formants

def sff(data, winsize, sample_rate):
    D = abs(librosa.core.stft(data,n_fft = winsize,  hop_length = winsize, win_length = winsize, center = False))
    freqs = librosa.fft_frequencies(sr = 22050, n_fft = winsize)
    fmax = np.amax(D,axis = 0)
    fcenter = np.percentile(D, 50, axis = 0)
    fmean = []
    newd  = D.transpose()
    fmean1 = []
    fmean2 = []
    fmean3 = []
    fmean4 = []
    fmean5 = []
    fmean6 = []
    fmean7 = []
    fmean8 = []
    for d in newd:
        fmean.append(sum(d[:len(freqs[freqs < 8000])] * freqs[:len(freqs[freqs < 8000])]) / sum(d[:len(freqs[freqs < 8000])]))
        fmean1.append(sum(d[:len(freqs[freqs < 1000])] * freqs[:len(freqs[freqs < 1000])]) / sum(d[:len(freqs[freqs < 1000])]))
        fmean2.append(sum(d[len(freqs[freqs < 1000]):len(freqs[freqs < 2000])] * freqs[len(freqs[freqs < 1000]):len(freqs[freqs < 2000])]) / sum(d[len(freqs[freqs < 1000]):len(freqs[freqs < 2000])]))
        fmean3.append(sum(d[len(freqs[freqs < 2000]):len(freqs[freqs < 3000])] * freqs[len(freqs[freqs < 2000]):len(freqs[freqs < 3000])]) / sum(d[len(freqs[freqs < 2000]):len(freqs[freqs < 3000])]))
        fmean4.append(sum(d[len(freqs[freqs < 3000]):len(freqs[freqs < 4000])] * freqs[len(freqs[freqs < 3000]):len(freqs[freqs < 4000])]) / sum(d[len(freqs[freqs < 3000]):len(freqs[freqs < 4000])]))
        fmean5.append(sum(d[len(freqs[freqs < 4000]):len(freqs[freqs < 5000])] * freqs[len(freqs[freqs < 4000]):len(freqs[freqs < 5000])]) / sum(d[len(freqs[freqs < 4000]):len(freqs[freqs < 5000])]))
        fmean6.append(sum(d[len(freqs[freqs < 5000]):len(freqs[freqs < 6000])] * freqs[len(freqs[freqs < 5000]):len(freqs[freqs < 6000])]) / sum(d[len(freqs[freqs < 5000]):len(freqs[freqs < 6000])]))
        fmean7.append(sum(d[len(freqs[freqs < 6000]):len(freqs[freqs < 7000])] * freqs[len(freqs[freqs < 6000]):len(freqs[freqs < 7000])]) / sum(d[len(freqs[freqs < 6000]):len(freqs[freqs < 7000])]))
        fmean8.append(sum(d[len(freqs[freqs < 7000]):len(freqs[freqs < 8000])] * freqs[len(freqs[freqs < 7000]):len(freqs[freqs < 8000])]) / sum(d[len(freqs[freqs < 7000]):len(freqs[freqs < 8000])]))
    fmean = np.array(fmean)
    fmean1 = np.array(fmean1)
    fmean2 = np.array(fmean2)
    fmean3 = np.array(fmean3)
    fmean4 = np.array(fmean4)
    fmean5 = np.array(fmean5)
    fmean6 = np.array(fmean6)
    fmean7 = np.array(fmean7)
    fmean8 = np.array(fmean8)
    SFF = np.vstack([fmax, fcenter, fmean, fmean1, fmean2, fmean3, fmean4, fmean5, fmean6, fmean7, fmean8])
    return SFF.transpose()
    
def ser(data, winsize, sample_rate):
    D = abs(librosa.core.stft(data,n_fft = winsize,  hop_length = winsize, win_length = winsize, center = False))
    freqs = librosa.fft_frequencies(sr = 22050, n_fft = winsize)
    newd  = D.transpose()
    ser1 = []
    ser2 = []
    ser3 = []
    ser4 = []
    ser5 = []
    ser6 = []
    ser7 = []
    ser8 = []
    for d in newd:
        ser1.append(sum(np.square(d[:len(freqs[freqs < 1000])])) / sum(d[:len(freqs[freqs < 1000])]))
        ser2.append(sum(np.square(d[len(freqs[freqs < 1000]):len(freqs[freqs < 2000])])) / sum(np.square(d[len(freqs[freqs < 1000]):len(freqs[freqs < 2000])])))
        ser3.append(sum(np.square(d[len(freqs[freqs < 2000]):len(freqs[freqs < 3000])])) / sum(np.square(d[len(freqs[freqs < 2000]):len(freqs[freqs < 3000])])))
        ser4.append(sum(np.square(d[len(freqs[freqs < 3000]):len(freqs[freqs < 4000])])) / sum(np.square(d[len(freqs[freqs < 3000]):len(freqs[freqs < 4000])])))
        ser5.append(sum(np.square(d[len(freqs[freqs < 4000]):len(freqs[freqs < 5000])])) / sum(np.square(d[len(freqs[freqs < 4000]):len(freqs[freqs < 5000])])))
        ser6.append(sum(np.square(d[len(freqs[freqs < 5000]):len(freqs[freqs < 6000])])) / sum(np.square(d[len(freqs[freqs < 5000]):len(freqs[freqs < 6000])])))
        ser7.append(sum(np.square(d[len(freqs[freqs < 6000]):len(freqs[freqs < 7000])])) / sum(np.square(d[len(freqs[freqs < 6000]):len(freqs[freqs < 7000])])))
        ser8.append(sum(np.square(d[len(freqs[freqs < 7000]):len(freqs[freqs < 8000])])) / sum(np.square(d[len(freqs[freqs < 7000]):len(freqs[freqs < 8000])])))

    ser1 = np.array(ser1)
    ser2 = np.array(ser2)
    ser3 = np.array(ser3)
    ser4 = np.array(ser4)
    ser5 = np.array(ser5)
    ser6 = np.array(ser6)
    ser7 = np.array(ser7)
    ser8 = np.array(ser8)
    SER = np.vstack([ser1, ser2, ser3, ser4, ser5, ser6, ser7, ser8])
    return SER.transpose()

def power_ratio(data, winsize, sample_rate):
    D = abs(librosa.core.stft(data,n_fft = winsize,  hop_length = winsize, win_length = winsize, center = False))
    freqs = librosa.fft_frequencies(sr = 22050, n_fft = winsize)
    newd  = D.transpose()
    pr = []
    for d in newd:
        pr.append(math.log(sum(np.square(d[:len(freqs[freqs < 800])])) / sum(np.square(d[len(freqs[freqs < 800]):len(freqs[freqs < 8000])]))))
    return np.array(pr)
        
def mfcc(data, winsize, sr ):
    D = np.abs(librosa.core.stft(data,n_fft = winsize,  hop_length = winsize, win_length = winsize, center = False))**2
    S = librosa.feature.melspectrogram(S=D, y=data)
    feats = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
    return feats.transpose()

def WPT(data, winsize):
    data_matrix = librosa.util.frame(data, winsize, winsize, axis = 0)
    WPTE = []
    for frame in data_matrix:
        wp = pywt.WaveletPacket(data = frame, wavelet = 'sym3', maxlevel = 7)
        paths = []
        wpte = []
        wte =[]
        for i in range(8):
            path = [node.path for node in wp.get_level(i)]
            paths = paths + path
        for path in paths:
            wpte.append(math.sqrt(sum(np.square(wp[path].data)) / wp[path].data.size))
        
        WPTE.append(wpte)
    WPTE = np.array(WPTE)
    return WPTE
            
#I still don't know what they want with EMDF
def EMDF(data, winsize):
    data_matrix = librosa.util.frame(data, winsize, winsize, axis = 0)
    for frame in data_matrix:
        emd = EMD()
        IMFs = emd.emd(frame)
        print(IMFs.shape)


if __name__ == "__main__":
    folders = ['1-tag ok']
    win_size = 0.02
    
    
    audiofiles = []
    sample_rates = []
    for folder in folders:
        filenames = []
        names = read_folder(folder)
        filenames = filenames + names
        
        for filename in filenames:
            audio, sr = read_wav(folder + '/' + filename)
            audiofiles.append(audio)
            sample_rates.append(sr)
            
            
    crest = []
    formant = []
    for i, audio in enumerate(audiofiles):
        winsize = math.floor(sample_rates[i] * win_size)
        pr = power_ratio(audio, winsize, sample_rates[i])
        print(pr.shape)
        
        