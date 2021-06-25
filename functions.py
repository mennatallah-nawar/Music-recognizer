import librosa
import librosa.display
from scipy import signal
from PIL import Image
import pylab
import imagehash
from imagehash import hex_to_hash
import json


def creat_dic(filename):
    HashSongDict = {
        filename: {#"spectrogram Hash": None,
        "centroid Hash": None,
        "rolloff Hash": None,
        "chroma stft Hash": None}
    }
    return HashSongDict

def save_dic(filename, features_array):

    dic = creat_dic(filename)
    #dic[filename]["spectrogram Hash"] = Hash(Data)
    dic[filename]['centroid Hash'] = Hash(features_array[0])
    dic[filename]['rolloff Hash'] = Hash(features_array[1])
    dic[filename]['chroma stft Hash'] = Hash(features_array[2])
    return dic

def read_file(path):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    for song in data:
        yield song, data[song]

def read_Wav (FilePath):
    wavsong,samplingFrequency = librosa.load(FilePath,duration=60 )
    filename = FilePath.split("/")[-1]
    filename = filename.split(".")[0]
    return wavsong,samplingFrequency,filename

def spectrogram(audio, fs, filename):
    f, t, Data = signal.spectrogram(audio, fs=fs, window='hann')   
    spectro_image = Image.fromarray(Data, mode='RGB')
    spectro_image.save("wav/spectro_features/"+filename+"_spectro.png")
    return Data

def extract_features(filename, audio, fs, spectrogram):

    # centroid feature
    CentroidSavePath = 'wav/Features/'+filename+'_centroid.png'
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    centroid = librosa.feature.spectral_centroid(y = audio, sr = fs, S = spectrogram)
    librosa.display.specshow(centroid,sr=fs)
    pylab.savefig(CentroidSavePath, bbox_inches=None, pad_inches=0)
    pylab.close()

    # rolloff feature
    RolloffSavePath = 'wav/Features/'+filename+'_rolloff.png'
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    rolloff = librosa.feature.spectral_rolloff(y = audio, sr = fs, S = spectrogram)
    librosa.display.specshow(rolloff,sr=fs)
    pylab.savefig(RolloffSavePath, bbox_inches=None, pad_inches=0)
    pylab.close()

    #chroma stft
    ChromaSavePath = 'wav/Features/'+filename+'_chroma_stft.png'
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    chroma_stft  = librosa.feature.chroma_stft(y = audio, sr = fs, S = spectrogram)
    librosa.display.specshow(chroma_stft,sr=fs)
    pylab.savefig(ChromaSavePath, bbox_inches=None, pad_inches=0)
    pylab.close()

    return[centroid,rolloff,chroma_stft]

def Hash(feature):
    data = Image.fromarray(feature)
    return imagehash.phash(data, hash_size = 16).__str__()

# Hamming is used to compare two hashes and export the differencies
def Hamming(hash1, hash2):
    similarity = 1 - ( hex_to_hash(hash1) - hex_to_hash(hash2) )/256.0
    return similarity