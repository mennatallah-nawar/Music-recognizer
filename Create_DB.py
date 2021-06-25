import os
from textwrap import indent
from functions import *
import json


#Iterate files inside given directory

def DB_Iteration ():
    filedb = {}
    directory = r'/media/tarek/New Volume1/Yomna/Shazam/yomna/wav'
    for entry in os.scandir(directory):
        if (entry.path.endswith(".wav")) and entry.is_file():
            print(entry.path)
            
            wavsong,samplingFrequency,filename = read_Wav (entry.path)

                                ## Create Spectrogram ##

            Data = spectrogram(wavsong, samplingFrequency, filename)

                                    ## Features ##

            features_array = extract_features(filename, wavsong, samplingFrequency, Data)

                                ## create dictionary with Hashing ##   

            dic = save_dic(filename, features_array)

                                    ## add dictionary to database ##
            filedb.update(dic)
    with open('wav/DB.txt', 'a') as file:
        json.dump(filedb, file, indent=3)

            



DB_Iteration()
