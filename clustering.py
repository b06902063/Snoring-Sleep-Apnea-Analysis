from feature_extract import read_folder, read_wav, crest_factor, get_formants, sff, ser, power_ratio, mfcc, WPT
from sklearn.cluster import KMeans
import numpy as np
import math
if __name__ == "__main__":
    folders = ['1-tag ok', '2-tag ok', '3-tag ok', '4-tag ok', '5-tag ok', '6-tag ok', '7-tag ok', '8-tag ok', '9-tag ok']
    win_size = 0.02
    
    
    #read data
    
    audiofiles = [] #audio samples
    sample_rates = [] #sampling rate of each sample
    for folder in folders:
        filenames = []
        names = read_folder(folder)
        filenames = filenames + names
        
        for filename in filenames:
            audio, sr = read_wav(folder + '/' + filename)
            audiofiles.append(audio)
            sample_rates.append(sr)
            
    
    #features
    Crest = np.load('crest.npy') #crest factor
    Formants = [] #formants
    SFF = np.load('sff.npy') #spectral frequency features
    SER = np.load('ser.npy') #subband energy ratio
    PR = np.load('pr.npy') #power ratio
    MFCC = np.load('mfcc.npy') #mel-scale frequency cepstrum coefficients
    Wpt = np.load('wpt.npy') #wavelet energy features
    '''
    for i, audio in enumerate(audiofiles):
        print(i, 'th file in ', len(audiofiles), ' files')
        winsize = math.floor(sample_rates[i] * win_size)
        cresti = crest_factor(audio, winsize)
        sffi = sff(audio, winsize, sample_rates[i])
        seri = ser(audio, winsize, sample_rates[i])
        pri = power_ratio(audio, winsize, sample_rates[i])
        mfcci = mfcc(audio, winsize, sample_rates[i])
        wpti = WPT(audio, winsize)
        
        print(cresti.shape)
        print(sffi.shape)
        print(seri.shape)
        print(pri.shape)
        print(mfcci.shape)
        print(wpti.shape)
        
        Crest.append(cresti)
        SFF.append(sffi)
        SER.append(seri)
        PR.append(pri)
        MFCC.append(mfcci)
        Wpt.append(wpti)
        
    Crest = np.array(Crest)
    SFF = np.array(SFF)
    SER = np.array(SER)
    PR = np.array(PR)
    MFCC = np.array(MFCC)
    Wpt = np.array(Wpt)
    
    np.save('crest.npy', Crest)
    np.save('sff.npy', SFF)
    np.save('ser.npy', SER)
    np.save('pr.npy', PR)
    np.save('mfcc.npy', MFCC)
    np.save('wpt.npy', Wpt)
    '''
    Crest = Crest[:,:,np.newaxis]
    PR = PR[:,:,np.newaxis]
    All_Features = np.dstack([Crest,SFF,SER,PR,MFCC,Wpt])
    np.save('features.npy', All_Features)
    
    kmeans = KMeans(init = 'k-means++', n_clusters = 3)
    
    
    All_Features[np.isnan(All_Features) == True] = 0
    All_Features[np.isfinite(All_Features) == False] = np.finfo('float64').max
    kmeans.fit(All_Features.reshape(-1,289))
    
    clusters = kmeans.labels_
    print(clusters.shape)
    
    np.save('clusters.npy', clusters)