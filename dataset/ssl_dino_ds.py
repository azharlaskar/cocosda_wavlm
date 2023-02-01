


from dataset.sampler import HierarchicalSampler
import torch 
import numpy as np 
import random 
import os,sys,json,glob
import soundfile
import numpy as np 
import soundfile
import random
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,OneOf
import numpy as np
import torch
from audiomentations import AddBackgroundNoise, PolarityInversion,ApplyImpulseResponse,Gain,TimeStretch
def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats,axis=0).astype(np.float)
    return feat
class SpeakerDataset(object):
    def __init__(self, path_to_file_train, transform):
        self.path_to_file_train = path_to_file_train
        dict_speaker={}
        speaker,paths,regions,addtions=[],[],[],[] 
        with open (self.path_to_file_train,"r") as f:
            for line in f.readlines():
                line= str(line.split()[0])
                if line not in dict_speaker:
                    dict_speaker[line] = len(dict_speaker)
        self.dict_speaker=dict_speaker
        with open(self.path_to_file_train,"r") as f:
            for line in f.readlines():
                line = line.split()
                    
                speaker.append(
                    dict_speaker[line[0]]
                )
                paths.append(
                    line[1]
                )
                regions.append(
                    line[2]
                )
                if len(line) > 3:
                    addtions.append(line[3:])
        self.speaker=speaker
        self.paths=paths
        self.dict_region={}
        for i in regions:
            if i not in self.dict_region:
                self.dict_region[i] = len(self.dict_region)
            
        self.regions=[self.dict_region[i] for i in regions]
        self.addtions=addtions

        self.transform = transform

    def __len__(self):return len(self.speaker)

    def __getitem__(self, idx):
        path=self.paths[idx]
        label = self.speaker[idx] 
        regions=self.regions[idx]
        label=(label,regions)
        wav, sr = torchaudio.load(path)
        wav = self.transform(wav)
        return wav, label 