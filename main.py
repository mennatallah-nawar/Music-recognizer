#import the pyplot and wavfile modules 
from PyQt5 import QtWidgets,QtCore,QtGui
import os
from os import path ## os --> Operating system / path --> Ui File
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
#from gui import Ui_MainWindow
import os
import sys
import matplotlib.pyplot as plot
import librosa 
from pydub import AudioSegment
from tempfile import mktemp
#import sklearn
import librosa.display
import numpy as np
from PIL import Image
import imagehash 
import pylab
import matplotlib.pyplot as plt

from functions import *


FORM_CLASS,_= loadUiType(path.join(path.dirname(__file__),"gui.ui"))
class MainApp(QtWidgets.QMainWindow, FORM_CLASS):                # QmainWindow: refers to main window in Qt Designer
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.song1 = 0
        self.song2 = 0
        self.fs1 = 0
        self.fs2 = 0
        self.Mixer_Flag = False
        self.Compare_Flag = False
        self.songs = [self.song1, self.song2]
        self.fs = [self.fs1, self.fs2]
        self.song_browse_flag = 0
        self.features_array = [0,0,0]
        self.similarityResults = []
        self.file = {}
        self.Text_Label = [self.label, self.label_2]
        self.Handel_Buttons()


    def Handel_Buttons(self):
        self.Song_1.clicked.connect(lambda: self.browseSong(0))
        self.Song_2.clicked.connect(lambda: self.browseSong(1))
        self.Mixer.clicked.connect(self.flag_on)
        self.Song.clicked.connect(self.flag_off)
        self.horizontalSlider.valueChanged.connect(self.mixer)
    
    def flag_on(self):
        self.Mixer_Flag = True
    def flag_off(self):
        self.Mixer_Flag = False


    def browseSong(self, song_number):
        filepath = QtWidgets.QFileDialog.getOpenFileName(self, 'choose song', os.getenv('HOME') ,"wav(*.wav)")
        self.Compare_Flag = True

        if self.Mixer_Flag == True:
            self.songs[song_number], self.fs[song_number], self.filename = read_Wav (filepath[0])
            self.Text_Label[song_number].setText(self.filename)
            self.song_browse_flag += 1
            self.mixer()
        else:
            wavsong, samplingFrequency, self.filename = read_Wav (filepath[0])
            self.Text_Label[song_number].setText(self.filename)
            Data = spectrogram(wavsong, samplingFrequency, self.filename)
            self.features_array = extract_features(self.filename, wavsong, samplingFrequency, Data)
            self.compare()

        
    def mixer(self):
        if self.song_browse_flag == 2:
            slider_val = self.horizontalSlider.value()/100
            self.outputsong = (self.songs[0] * slider_val) + (self.songs[1] * (1 - slider_val))
            Data = spectrogram(self.outputsong, self.fs[0], self.filename)
            self.features_array = extract_features(self.filename, self.outputsong, self.fs[0], Data)
            self.compare()

            
    def compare(self):
        if self.Compare_Flag == True:
            file_hash = {}
            for songName, songHashes in read_file("wav/DB.txt"):

                centroidHamming = Hamming(songHashes['centroid Hash'], Hash(self.features_array[0]))
                rolloffHamming = Hamming(songHashes['rolloff Hash'], Hash(self.features_array[1]))
                chromaHamming = Hamming(songHashes['chroma stft Hash'], Hash(self.features_array[2]))
    
                output = (centroidHamming + rolloffHamming + chromaHamming)/3
                self.similarityResults.append((songName , output*100))
            
            dic = save_dic(self.filename, self.features_array)
            file_hash.update(dic)
            with open('wav/File.txt', 'a') as file:
                json.dump(file_hash, file, indent=4)

            self.similarityResults.sort(key= lambda x: x[1], reverse=True)
            self.fill_table()

    def fill_table(self):

        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Similarity", "Percentage"])

        for row in range(10):
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(self.similarityResults[row][0]))
            self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(str(round(self.similarityResults[row][1], 2))+"%"))

        for col in range(2):
            self.tableWidget.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)
            self.tableWidget.horizontalHeaderItem(col).setBackground(QtGui.QColor(57, 65, 67))
        self.similarityResults.clear()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_() #infinite loop

if __name__=='__main__':
    main()
