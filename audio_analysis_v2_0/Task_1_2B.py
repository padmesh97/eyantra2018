
## Mocking Bot - Task 1.2 B: Detection of Notes and Silence Duration

#  Instructions
#  ------------
#
#  This file contains Main function, detect_note_duration function and detect_silence_duration function.
#  Main Function helps you to check your output for practice audio files provided.
#  Do not make any changes in the Main Function. You have to complete only the
#  detect_note_duration function and detect_silence_duration function. You can add helper functions
#  but make sure that these functions are called from functions provided. The final output
#  should be returned from the detect_note_duration function and detect_silence_duration function.
#
#  Note: While evaluation we will use only the detect_note_duration function and detect_silence_duration function.
#  Hence the format of input, output or returned arguments should be as per the given format.
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
from scipy import signal

# Teams can add helper functions
# Add all helper functions here

############################### Your Code Here #############################################

rms=[]
sil=[]
ton=[]

def detect_note_duration(audio_file):
        
        #       Instructions
        #       ------------
        #       Input   :       audio_file -- a single test audio_file as input argument
        #       Output  :       Note_durations -- List of list containing Float numbers corresponding
        #                                                                 to the note start time and end time
        #                                                                 (up to Two decimal places)
        #       Example :       For Audio_1.wav file,
        #                               Note_durations = [[0.00, 0.610], [0.86, 0.99], [1.28, 1.48], [1.60, 1.80], [2.05, 2.82]]

        Note_durations = []

        # Add your code here
        rms=[]
        dur=[]
        silent=0
        file_length=audio_file.getnframes()
        sound=np.zeros(file_length)
        for i in range(file_length):
                data=audio_file.readframes(1)
                data=struct.unpack("<h",data)
                sound[i]=int(data[0])
        
        y=signal.unit_impulse(1)

        win_size=int(0.01*audio_file.getframerate())
        
        for i in range(0,sound.size,int(win_size/2)):
                if i==0:
                        c=signal.fftconvolve(sound[i:(i+int(2*win_size))],y,mode='full')
                        rms.append(np.sum(np.square(c)))
                else:
                        c=signal.fftconvolve(sound[i:(i+win_size)],y,mode='full')
                        rms.append(np.sum(np.square(c)))
                        
        j=0
        thresh=0.1*math.pow(10,math.floor(math.log(max(rms),10)))
        silent=0              
        for i in range(0,sound.size,int(win_size/2)):
                if(rms[j]>=thresh and silent==0):
                        silent=1
                        ton.append(round(i/float(audio_file.getframerate()),2))
                        
                if(rms[j]<thresh and silent==1):
                        silent=0
                        sil.append(round(i/float(audio_file.getframerate()),2))
                        

                j=j+1
                
        dur=[]

                        
        for i in range(0,len(ton)):
                dur.append(ton[i])
                dur.append(sil[i])
                if abs(dur[1]-dur[0])>=0.08:
                        Note_durations.append(dur)
                dur=[]
                
        return Note_durations


############################### Your Code Here #############################################

def detect_silence_duration(audio_file):
        
        #       Instructions
        #       ------------
        #       Input   :       audio_file -- a single test audio_file as input argument
        #       Output  :       Silence_durations -- List of list containing Float numbers corresponding
        #                                                                        to the silence start time and end time
        #                                                                        (up to Two decimal places)
        #       Example :       For Audio_1.wav file,
        #                               Silence_durations = [[0.61, 0.86], [0.99, 1.28], [1.48, 1.60], [1.80, 2.05]] 

        Silence_durations = []

        # Add your code here

        dur=[]
        sil=[]
        rms=[]
        for i in range(0,len(sil)-1):
                dur.append(sil[i])
                dur.append(ton[i+1])
                if abs(dur[1]-dur[0])>=0.06:
                        Silence_durations.append(dur)
                dur=[]
            
        return Silence_durations


############################### Main Function #############################################

if __name__ == "__main__":

        #   Instructions
        #   ------------
        #   Do not edit this function.

        # code for checking output for single audio file
        path = os.getcwd()
        
        file_name = path + "\Task_1.2B_Audio_files\Audio_1.wav"
        audio_file = wave.open(file_name)
        
        Note_durations = detect_note_duration(audio_file)
        Silence_durations = detect_silence_duration(audio_file)

        print("\n\tNotes Duration = " + str(Note_durations))
        print("\n\tSilence Duration = " + str(Silence_durations))

        # code for checking output for all audio files
        x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")

        if x == 'Y':

                Note_durations_list = []

                Silence_durations_list = []

                file_count = len(os.listdir(path + "\Task_1.2B_Audio_files"))

                for file_number in range(1, file_count):

                        file_name = path +"\Task_1.2B_Audio_files\Audio_"+str(file_number)+".wav"
                        audio_file = wave.open(file_name)
                        
                        Note_durations = detect_note_duration(audio_file)
                        Silence_durations = detect_silence_duration(audio_file)
                        
                        Note_durations_list.append(Note_durations)
                        Silence_durations_list.append(Silence_durations)

                print("\n\tNotes Duration = " + str(Note_durations_list[0]) + ",\n\t\t\t" + str(Note_durations_list[1]) + ",\n\t\t\t" + str(Note_durations_list[2]))
                print("\n\tSilence Duration = " + str(Silence_durations_list[0]) + ",\n\t\t\t" + str(Silence_durations_list[1]) + ",\n\t\t\t" + str(Silence_durations_list[2]))
                
