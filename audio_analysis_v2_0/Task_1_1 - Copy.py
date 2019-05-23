

## Mocking Bot - Task 1.1: Note Detection

#  Instructions
#  ------------
#
#  This file contains Main function and note_detect function. Main Function helps you to check your output
#  for practice audio files provided. Do not make any changes in the Main Function.
#  You have to complete only the note_detect function. You can add helper functions but make sure
#  that these functions are called from note_detect function. The final output should be returned
#  from the note_detect function.
#
#  Note: While evaluation we will use only the note_detect function. Hence the format of input, output
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

# Teams can add helper functions
# Add all helper functions here

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

def note_detect(audio_file):

        #   Instructions
        #   ------------
        #   Input   :   audio_file -- a single test audio_file as input argument
        #   Output  :   Detected_Note -- String corresponding to the Detected Note
        #   Example :   For Audio_1.wav file, Detected_Note = "A4"

        Detected_Note = ""

        # Add your code here
        file_length=audio_file.getnframes()
        sound=np.zeros(file_length)
        for i in range(file_length):
                data=audio_file.readframes(1)
                data=struct.unpack("<h",data)
                sound[i]=int(data[0])

        ft=np.fft.fft(sound)
        """g=abs(ft)
        m=np.argmax(g)
        det_freq=m*audio_file.getframerate()/sound.size"""

        phas=np.angle(ft)

        pd=peakDetection(ft,math.pow(10,math.floor(math.log(max(ft.real),10))))
        h=pd[0]*audio_file.getframerate()/sound.size

        
        freq,ipmag,ipphase=peakInterp(abs(ft),np.angle(ft),pd)
        #print freq
        det_freq=freq[0]*audio_file.getframerate()/sound.size
        print pd,sound.size
        
        sto_freq=[16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,51.91,55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,87.31,92.50,98.00,103.83,110.00,116.54,123.47,130.81,138.59,146.83,155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392.00,415.30,440.00,466.16,493.88,523.25,554.37,587.33,622.25,659.25,698.46,739.99,783.99,830.61,880.00,932.33,987.77,1046.50,1108.73,1174.66,1244.51,1318.51,1396.91,1479.98,1567.98,1661.22,1760.00,1864.66,1975.53,2093.00,2217.46,2349.32,2489.02,2637.02,2793.83,2959.96,3135.96,3322.44,3520.00,3729.31,3951.07,4186.01,4434.92,4698.63,4978.03,5274.04,5587.65,5919.91,6271.93,6644.88,7040.00,458.62,7902.13]
        sto_notes=['C0','C#0','D0','D#0 ','E0','F0','F#0','G0','G#0','A0','A#0','B0','C1','C#1 ','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1','C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5','C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6','C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7','C8','C#8','D8','D#8','E8','F8','F#8','G8','G#8','A8','A#8','B8']
        plt.plot(ft)
        #plt.plot(freq,ipmag,marker='x')
        plt.show()


        for k in range(len(sto_freq)):
                if abs(det_freq-sto_freq[k])<=10:
                        Detected_Note=sto_notes[k]
                        break
        
        return Detected_Note


############################### Main Function ##############################################

if __name__ == "__main__":

        #   Instructions
        #   ------------
        #   Do not edit this function.

        # code for checking output for single audio file
        path = os.getcwd()
        
        file_name = path + "\Task_1.1_Audio_files\Audio_1.wav"
        audio_file = wave.open(file_name)

        Detected_Note = note_detect(audio_file)

        print("\n\tDetected Note = " + str(Detected_Note))

        # code for checking output for all audio files
        x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")
        
        if x == 'Y':

                Detected_Note_list = []

                file_count = len(os.listdir(path + "\Task_1.1_Audio_files"))

                for file_number in range(1, file_count):

                        file_name = path + "\Task_1.1_Audio_files\Audio_"+str(file_number)+".wav"
                        audio_file = wave.open(file_name)

                        Detected_Note = note_detect(audio_file)
                        
                        Detected_Note_list.append(Detected_Note)

                print("\n\tDetected Notes = " + str(Detected_Note_list))
        
        
