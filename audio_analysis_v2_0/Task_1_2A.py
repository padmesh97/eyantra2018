
## Mocking Bot - Task 1.2 A: Notes and Onset Detection

#  Instructions
#  ------------
#
#  This file contains Main function and onset_detect function. Main Function helps you to check your output
#  for practice audio files provided. Do not make any changes in the Main Function.
#  You have to complete only the onset_detect function. You can add helper functions but make sure
#  that these functions are called from onset_detect function. The final output should be returned
#  from the onset_detect function.
#
#  Note: While evaluation we will use only the onset_detect function. Hence the format of input, output
#  or returned arguments should be as per the given format.
#  
#  Recommended Python version is 2.7.
#  The submitted Python file must be 2.7 compatible as the evaluation will be done on Python 2.7.
#  
#  Warning: The error due to compatibility will not be entertained.
#  -------------


## Library initialisation

# Import Modules
# DO NOT import any library/module
# related to Audio Processing here
import numpy as np
import math
import wave
import os
import struct
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft, ifft

# Teams can add helper functions
# Add all helper functions here


def TWM_p(pfreq, pmag, f0c):
	"""
	Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
	[better to use the C version of this function: UF_C.twm]
	pfreq, pmag: peak frequencies in Hz and magnitudes,
	f0c: frequencies of f0 candidates
	returns f0, f0Error: fundamental frequency detected and its error
	"""

	p = 0.5                                          # weighting by frequency value
	q = 1.4                                          # weighting related to magnitude of peaks
	r = 0.5                                          # scaling related to magnitude of peaks
	rho = 0.33                                       # weighting of MP error
	Amax = max(pmag)                                 # maximum peak magnitude
	maxnpeaks = 10                                   # maximum number of peaks used
	harmonic = np.matrix(f0c)
	ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
	MaxNPM = min(maxnpeaks, pfreq.size)
	for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
		difmatrixPM = harmonic.T * np.ones(pfreq.size)
		difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
		FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
		peakloc = np.argmin(difmatrixPM, axis=1)
		Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
		PeakMag = pmag[peakloc]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
		harmonic = harmonic+f0c

	ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
	MaxNMP = min(maxnpeaks, pfreq.size)
	for i in range(0, f0c.size) :                    # measured to predicted mismatch error
		nharm = np.round(pfreq[:MaxNMP]/f0c[i])
		nharm = (nharm>=1)*nharm + (nharm<1)
		FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
		Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
		PeakMag = pmag[:MaxNMP]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

	Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
	f0index = np.argmin(Error)                       # get the smallest error
	f0 = f0c[f0index]                                # f0 with the smallest error

	return f0, Error[f0index]

def isPower2(num):
	"""
	Check if num is power of two
	"""
	return ((num & (num - 1)) == 0) and num > 0

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}


def dftAnal(x, w, N):
	"""
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size 
	returns mX, pX: magnitude and phase spectrum
	"""

	if not(isPower2(N)):                                 # raise error if N not a power of two
		raise ValueError("FFT size (N) is not a power of 2")

	if (w.size > N):                                        # raise error if window size bigger than fft size
		raise ValueError("Window size (M) is bigger than FFT size")

	hN = (N//2)+1                                           # size of positive spectrum, it includes sample 0
	hM1 = (w.size+1)//2                                     # half analysis window size by rounding
	hM2 = w.size//2                                         # half analysis window size by floor
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	xw = x*w                                                # window the input sound
	fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
	fftbuffer[-hM2:] = xw[:hM2]        
	X = fft(fftbuffer)                                      # compute FFT
	absX = abs(X[:hN])                                      # compute ansolute value of positive side
	absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
	mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
	tol = 1e-14
	X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
	X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values         
	pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
	return mX, pX


def peakInterp(mX, pX, ploc):
	"""
	Interpolate peak values using parabolic interpolation
	mX, pX: magnitude and phase spectrum, ploc: locations of peaks
	returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
	"""

	val = mX[ploc]                                          # magnitude of peak bin
	lval = mX[ploc-1]                                       # magnitude of bin at left
	rval = mX[ploc+1]                                       # magnitude of bin at right
	iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
	ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
	ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
	return iploc,ipmag,ipphase


def peakDetection(mX, t):
        """
        Detect spectral peak locations
        mX: magnitude spectrum, t: threshold
        returns ploc: peak locations
        """

        thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0); # locations above threshold
        next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
        prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
        ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
        ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
        return ploc

############################### Your Code Here ##############################################

def note_detect(sound):

        #   Instructions
        #   ------------
        #   Input   :   audio_file -- a single test audio_file as input argument
        #   Output  :   Detected_Note -- String corresponding to the Detected Note
        #   Example :   For Audio_1.wav file, Detected_Note = "A4"

        Detected_Note = ""

        # Add your code here
        """"file_length=audio_file.getnframes()
        sound=np.zeros(file_length)
        for i in range(file_length):
                data=audio_file.readframes(1)
                data=struct.unpack("<h",data)
                sound[i]=int(data[0])"""

        N=1024
        M=601
        t=-60
        minf0=16
        maxf0=7923
        hn=N/2
        hm=(M+1)/2
        w=get_window('hamming',M)
        fs=int(audio_file.getframerate())
        start=int(.8*fs)
        x1=sound[start:start+M]
        mx,px=dftAnal(x1,w,N)
        ploc=peakDetection(mx,t)
        iploc,ipmag,ipphase=peakInterp(mx,px,ploc)
        ipfreq=fs*iploc/N;
        f0c=np.argwhere((ipfreq>minf0) & (ipfreq<maxf0))[:,0]
        f0cf=ipfreq[f0c]
        f0,errors=TWM_p(ipfreq, ipmag, f0cf)
        sto_freq=[16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,51.91,55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,87.31,92.50,98.00,103.83,110.00,116.54,123.47,130.81,138.59,146.83,155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392.00,415.30,440.00,466.16,493.88,523.25,554.37,587.33,622.25,659.25,698.46,739.99,783.99,830.61,880.00,932.33,987.77,1046.50,1108.73,1174.66,1244.51,1318.51,1396.91,1479.98,1567.98,1661.22,1760.00,1864.66,1975.53,2093.00,2217.46,2349.32,2489.02,2637.02,2793.83,2959.96,3135.96,3322.44,3520.00,3729.31,3951.07,4186.01,4434.92,4698.63,4978.03,5274.04,5587.65,5919.91,6271.93,6644.88,7040.00,458.62,7902.13]
        sto_notes=['C0','C#0','D0','D#0 ','E0','F0','F#0','G0','G#0','A0','A#0','B0','C1','C#1 ','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1','C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5','C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6','C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7','C8','C#8','D8','D#8','E8','F8','F#8','G8','G#8','A8','A#8','B8']
        for k in range(len(sto_freq)):
                if abs(f0-sto_freq[k])<5:
                        Detected_Note=sto_notes[k]
                        break
        return Detected_Note


############################### Your Code Here #############################################

def onset_detect(audio_file):
        
        #   Instructions
        #   ------------
        #   Input       :       audio_file -- a single test audio_file as input argument
        #   Output      :       1. Onsets -- List of Float numbers corresponding
        #                                                        to the Note Onsets (up to Two decimal places)
        #                               2. Detected_Notes -- List of string corresponding
        #                                                                        to the Detected Notes
        #       Example :       For Audio_1.wav file,
        #                               Onsets = [0.00, 2.15, 4.30, 7.55]
        #                               Detected_Notes = ["F4", "B3", "C6", "A4"]

        Onsets = []
        Detected_Notes = []

        # Add your code here
        rms=[]
        sil=[]
        ton=[]
        silent=0
        file_length=audio_file.getnframes()
        sound=np.zeros(file_length)
        for i in range(file_length):
                data=audio_file.readframes(1)
                data=struct.unpack("<h",data)
                sound[i]=int(data[0])
        
        """y=signal.unit_impulse(1)
        print sound.size
        win_size=int(0.05*audio_file.getframerate())
        
        for i in range(0,sound.size,int(win_size/2)):
                if i==0:
                        c=signal.fftconvolve(sound[i:(i+(2*win_size))],y,mode='full')
                        rms.append(np.sum(np.square(c)))
                        print np.sum(np.square(c))
                else:
                        c=signal.fftconvolve(sound[i:(i+win_size)],y,mode='full')
                        rms.append(np.sum(np.square(c)))
        j=0
        
        for i in range(0,sound.size,int(win_size/2)):
                if(rms[j]>=0.1*math.pow(10,math.floor(math.log(max(rms),10))) and silent==0):
                        silent=1
                        if(i==0 or len(ton)==0):
                                ton.append(round(i/float(audio_file.getframerate()),2))
                        if(i<>0 and round(i/float(audio_file.getframerate()),2)-ton[len(ton)-1]>0.05):
                                ton.append(round(i/float(audio_file.getframerate()),2))
                        
                if(rms[j]<0.1*math.pow(10,math.floor(math.log(max(rms),10))) and silent==1):
                        silent=0
                        if(i==0 or len(sil)==0):
                                sil.append(round(i/float(audio_file.getframerate()),2))
                        if(i<>0 and round(i/float(audio_file.getframerate()),2)-sil[len(sil)-1]>0.05):
                                sil.append(round(i/float(audio_file.getframerate()),2))
                        
                j=j+1

        print sil
        print ton"""
        #print sound[94374]

        #plt.figure(1)
        #plt.plot(abs(sound))
        """plt.figure(2)
        plt.plot(sound)"""
        #plt.show()
        print sound.size
        clip=sound[88200:176400]
        st=note_detect(clip)
        print st
        
        return Onsets, Detected_Notes


############################### Main Function #############################################

if __name__ == "__main__":

        #   Instructions
        #   ------------
        #   Do not edit this function.

        # code for checking output for single audio file
        path = os.getcwd()
        
        file_name = path + "\Task_1.2A_Audio_files\Audio_1.wav"
        audio_file = wave.open(file_name)
        
        Onsets, Detected_Notes = onset_detect(audio_file)

        print("\n\tOnsets = " + str(Onsets))
        print("\n\tDetected Notes = " + str(Detected_Notes))

        # code for checking output for all audio files
        x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")
                
        if x == 'Y':

                Onsets_list = []
                Detected_Notes_list = []

                file_count = len(os.listdir(path + "\Task_1.2A_Audio_files"))

                for file_number in range(1, file_count):

                        file_name = path + "\Task_1.2A_Audio_files\Audio_"+str(file_number)+".wav"
                        audio_file = wave.open(file_name)

                        Onsets, Detected_Notes = onset_detect(audio_file)
                        
                        Onsets_list.append(Onsets)
                        Detected_Notes_list.append(Detected_Notes)

                print("\n\tOnsets = " + str(Onsets_list))
                print("\n\tDetected Notes = " + str(Detected_Notes_list))


