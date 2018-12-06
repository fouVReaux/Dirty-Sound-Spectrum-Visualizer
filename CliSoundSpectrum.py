#!/usr/bin/env python
##############################################
# Polito Torino 2018
# Auteur : FouVReaux
# Date de creation : 16/04/18
# Date de derniere modification : 16/04/18
##############################################
#------------------------------------------------------------------------------#
#Modules used in the class :                                                   #
#------------------------------------------------------------------------------#
#Library
import numpy as np
from scipy.signal import filtfilt
from numpy import nonzero, diff, mean 
import time
from recorder import SoundCardDataSource
import sys
import os

# Based on function from numpy 1.8
####################################################################################
#                               Level Module                                       #
####################################################################################
class Level_module:
    def __init__(self, recorder, fs):
        """
        The Level_module module
        =====================================

        This module is a Level (loudness) annalyzer. In Real Time process
        With it you can extract some basic features of the imput like : 
            - the Main Frequency of the imput (Real Time)
            - the Main bande (1/3 octave) of the imput (Real Time)
            - the avrage of all the Max freq (and the band)
            - the repartition of the sound (1/3 octave)

        Parameters
        ----------
        :param recorder: Module of the recorder
        :param fs: freq sampleling (can be : 12000, 22000, 44000)
        :type recorder: Object
        :type fs: int

        :Example:
        
        >>>FS = 44000
        >>>recorder = SoundCardDataSource(channels=1,num_chunks=3,sampling_rate=FS,chunk_size=4*1024)
        >>>
        >>>level_module = Level_module(recorder,FS)
        >>>

        Processing
        ----------
        compute_all_data(self)
        |   Compute all the data need to be run to refresh the data and to get some

        def avr_main_peak_freq_cal(self):
        |   Create the avr of all the peak in memory

        def avr_main_peak_bande_cal(self):
        |   return the band of the previous function

        
        
        Getting
        -------

            
        Printing
        ------
        prompt_all_features()
        |   print all the information extract by this module 

        .. Date:: 16/04/2018
        .. author:: Cyril Lavrat
        """
        #input vars
        self.recorder = recorder
        self.FS = fs

        #output global vars amplitude
        self.main_peak_freq = 0         # Value of the main peaks format int                    
        self.main_peak_band = 0         # Value of the main band associated to the main peak    
        self.avr_main_peak_freq = 0     # Avrage on 200 peak (int)                              
        self.avr_main_peak_band = 0     # Avrage on 200 peak (int)                               
        self.energy_by_band = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        # energy by band (band = 1/3 of octave)                 [T]
        
        #internal global vars
        self.size_last_peaks = 200      #Peaks info
        self.last_peaks = []            #Peaks info

        self.added_band = 0             #reset the sum for all the band (energy by band)
        self.amp_factor = 50  
    #--------------------------------
    def rfftfreq(self, n, d=1.0):
        """
        Return the Discrete Fourier Transform sample frequencies
        (for usage with rfft, irfft).

        The returned float array `f` contains the frequency bin centers in cycles
        per unit of the sample spacing (with zero at the start). For instance, if
        the sample spacing is in seconds, then the frequency unit is cycles/second.

        Given a window length `n` and a sample spacing `d`::

        f = [0, 1, ..., n/2-1, n/2] / (d*n) if n is even
        f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n) if n is odd

        Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
        the Nyquist frequency component is considered to be positive.

        Parameters
        ----------
        n : int
        Window length.
        d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

        Returns
        -------
        f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.
        """
        if not isinstance(n, int):
            raise ValueError("n should be an integer")
        val = 1.0/(n*d)
        N = n//2 + 1
        results = np.arange(0, N, dtype=int)
        return results * val

    def fft_slices(self, x):
        Nslices, Npts = x.shape
        window = np.hanning(Npts)

        # Calculate FFT
        fx = np.fft.rfft(window[np.newaxis, :] * x, axis=1)

        # Convert to normalised PSD
        Pxx = abs(fx)**2 / (np.abs(window)**2).sum()

        # Scale for one-sided (excluding DC and Nyquist frequencies)
        Pxx[:, 1:-1] *= 2

        # And scale by frequency to get a result in (dB/Hz)
        # Pxx /= Fs
        return Pxx ** 0.5

    def find_peaks(self, Pxx):
        # filter parameters
        b, a = [0.01], [1, -0.99]
        Pxx_smooth = filtfilt(b, a, abs(Pxx))
        peakedness = abs(Pxx) / Pxx_smooth

        # find peaky regions which are separated by more than 10 samples
        peaky_regions = nonzero(peakedness > 1)[0]
        edge_indices = nonzero(diff(peaky_regions) > 10)[0]  # RH edges of peaks
        edges = [0] + [(peaky_regions[i] + 5) for i in edge_indices]
        if len(edges) < 2:
            edges += [len(Pxx) - 1]

        peaks = []
        for i in range(len(edges) - 1):
            j, k = edges[i], edges[i+1]
            peaks.append(j + np.argmax(peakedness[j:k]))
        return peaks

    def fft_buffer(self, x):
        window = np.hanning(x.shape[0])

        # Calculate FFT
        fx = np.fft.rfft(window * x)

        # Convert to normalised PSD
        Pxx = abs(fx)**2 / (np.abs(window)**2).sum()

        # Scale for one-sided (excluding DC and Nyquist frequencies)
        Pxx[1:-1] *= 2

        # And scale by frequency to get a result in (dB/Hz)
        # Pxx /= Fs
        return Pxx ** 0.5
    #--------------------------------
    def compute_all_data(self):
        """
        """
        #get time and freq val from the sound card
        timeValues = self.recorder.timeValues
        freqValues = self.rfftfreq(len(timeValues),1./recorder.fs)

        #fft things
        data = self.recorder.get_buffer()
        weighting = np.exp(timeValues / timeValues[-1])
        Pxx = self.fft_buffer(weighting * data[:, 0])
        
        #--------------------------------------------------------------------<< Peak research
        #max peak research
        peaks = [p for p in self.find_peaks(Pxx) if Pxx[p] > 0.3]
        #Add this peak to last peak for the avrage
        for p in peaks:
            self.last_peaks.append(p)
            if (len(self.last_peaks) > self.size_last_peaks):
                self.last_peaks.pop(0) #keep memory of overflow
                
        self.main_peak_freq = []
        #if peaks is not empty 
        if peaks != []:
            #magic constance 3.7 : between 440/112 and 880/246 ;)
            self.main_peak_freq = int(max(peaks)*3.7)
        #set of the band of the main peak
        self.main_peak_band = int(self.get_name_third_Octave_Band_Filters(self.main_peak_freq))
        #--------------------------------------------------------------------<< energy by band
        #energy by band
        peaks = [p for p in self.find_peaks(Pxx) if Pxx[p] > 0.001]
        for p in peaks:
            if self.get_name_third_Octave_Band_Filters(p) != -1:
                self.energy_by_band[self.get_name_third_Octave_Band_Filters(p)]=self.energy_by_band[self.get_name_third_Octave_Band_Filters(p)]+Pxx[p]
                self.added_band = self.added_band + 1
                if self.added_band > self.size_last_peaks/2 :
                    self.energy_by_band = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    self.added_band = 0
                  
    def avr_main_peak_freq_cal(self):
        """
        """
        sum = 0
        if len(self.last_peaks) != 0:
            for i in range(len(self.last_peaks)):
                sum = sum + self.last_peaks[i]
            sum = 3.7*sum/len(self.last_peaks)
            self.avr_main_peak_freq = int(sum)
        return self.avr_main_peak_freq

    def avr_main_peak_bande_cal(self):
        """
        """
        self.avr_main_peak_freq_cal()
        return self.get_name_third_Octave_Band_Filters(self.avr_main_peak_freq)

    def Get_energy_by_band(self,band):
        """
        """
        return int(self.energy_by_band[band]*self.amp_factor)

    def get_name_third_Octave_Band_Filters(self, frequency):
        """
        Give the number of the Bande octave Filter 
        with a resolution of 1/3 of an octave
        """
        if frequency in range(14, 18):
            return 1
        if frequency in range(18, 22):
            return 2
        if frequency in range(22,28):
            return 3
        if frequency in range(28,28):
            return 4
        if frequency in range(28,36):
            return 5
        if frequency in range(36,45):
            return 6
        if frequency in range(45,57):
            return 7
        if frequency in range(57,71):
            return 8
        if frequency in range(71,89):
            return 9
        if frequency in range(89,112):
            return 10
        if frequency in range(112,140):
            return 11
        if frequency in range(140,180):
            return 12
        if frequency in range(180,224):
            return 13
        if frequency in range(224,280):
            return 14
        if frequency in range(280,355):
            return 15
        if frequency in range(355,450):
            return 16
        if frequency in range(450,540):
            return 17
        if frequency in range(540,710):
            return 18
        if frequency in range(710,900):
            return 19
        if frequency in range(900,1120):
            return 20
        if frequency in range(1120,1400):
            return 21
        if frequency in range(1400,1800):
            return 22
        if frequency in range(1800,2240):
            return 23
        if frequency in range(2240,2800):
            return 24
        if frequency in range(2800,3550):
            return 25
        if frequency in range(3550,4500):
            return 26
        if frequency in range(4500,5600):
            return 27
        if frequency in range(5600,7100):
            return 28
        if frequency in range(7100,9000):
            return 29
        if frequency in range(9000,11200):
            return 30
        if frequency in range(11200,14000):
            return 31
        if frequency in range(14000,18000):
            return 32
        if frequency in range(18000,24000):
            return 33
        else:
            return -1
    #--------------------------------
    def Get_main_peak_freq(self):
        """
        Give the freqency of the main peak
        """
        return self.main_peak_freq
    
    def Get_main_peak_band(self):
        """
        Give the avrage band of all the main pick
        """
        return self.main_peak_band

    def Get_avr_main_peak_freq(self):
        """
        Give the avrage frequency of all the main pick
        """
        return self.avr_main_peak_freq

    def Get_avr_main_peak_band(self):
        """
        Give the avrage band of all the main pick
        """
        return self.avr_main_peak_band

    def Get_energy_by_band(self,bande):
        """
        give the energy in a specific bande
        """
        return self.energy_by_band

    def prompt_all_features(self, number = False):
        """
        print all the informations about the audio flux in a terminal
        look like : 

        """
        os.system('clear')
        msg =       "=====================================================\n"
        msg = msg + "Main freq : " + str(self.main_peak_freq) + " (band : " + str(self.main_peak_band)+" )\n"
        msg = msg + "-----------------------------------------------------\n"
        msg = msg + "Avrage peaks frequency : " + str(self.avr_main_peak_freq_cal()) +" Hz (band : "
        msg = msg + str(self.avr_main_peak_bande_cal()) + ")\n"
        msg = msg + "-----------------------------------------------------\n"
        msg = msg + "1/3 octave viwer : \n"

        if number == True:
            for band in range(len(self.energy_by_band)):
                msg = msg + "(" +str(band) + "): " + str(int(self.energy_by_band[band]*self.amp_factor))+"\n"

        if number == False:
            for band in range(len(self.energy_by_band)):
                msg = msg + "(" +str(band) + "): " 
                for i in range(int(self.energy_by_band[band]*self.amp_factor)):
                    msg = msg + "@"
                msg = msg + "\n"
        print (msg)
        pass

####################################################################################

if __name__ == '__main__':
    #Setup recorder
    #FS = 12000
    #FS = 22000
    FS = 44000
    latency=0
    recorder = SoundCardDataSource(channels=1,num_chunks=3,sampling_rate=FS,chunk_size=4*1024)
    
    #Setup Level Module
    level_module = Level_module(recorder,FS)

    while 1:    
        level_module.compute_all_data()
        #level_module.avr_main_peak_freq_cal()
        #level_module.avr_main_peak_bande_cal()
        level_module.prompt_all_features()
        #time.sleep(latency)
