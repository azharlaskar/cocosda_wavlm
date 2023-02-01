from dataset.sampler import HierarchicalSampler
import torch 
import numpy as np 
import random 
import os,sys,json,glob
import soundfile
import numpy as np 
import soundfile
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,OneOf
import numpy as np
import torch
import numpy
import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
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

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

class TrainTransform():
    def __init__(self, args):
        
        self.max_frames=args.max_frames
        musan_files   = glob.glob(os.path.join(args.musan_path,'*/*/*/*.wav'))
        noise_musan =  glob.glob(os.path.join(args.musan_path,'noise/*/*/*.wav'))
        speech_musan =  glob.glob(os.path.join(args.musan_path,'speech/*/*/*.wav'))
        music_musan =  glob.glob(os.path.join(args.musan_path,'music/*/*/*.wav'))
        rirs=glob.glob(os.path.join(args.rir_path,'*/*/*.wav'))
        # self.instance_augment= Compose(
        #     [
        #         OneOf(
        #             [
        #                 Gain( min_gain_in_db=-6, max_gain_in_db=6,p=0.1),
        #                 TimeStretch(p=0.1)
        #             ],
        #             p=0.1
        #         ),
        #         OneOf( 
        #             [
                       
        #                 AddBackgroundNoise(
        #                     sounds_path=noise_musan,
        #                     min_snr_in_db=0,
        #                     max_snr_in_db=15,
        #                     noise_transform=PolarityInversion(),
        #                     p=1,
        #                 ),
        #                 AddBackgroundNoise(
        #                     sounds_path=speech_musan,
        #                     min_snr_in_db=13,
        #                     max_snr_in_db=20,
        #                     noise_transform=PolarityInversion(),
        #                     p=1,
        #                 ),
        #                 AddBackgroundNoise(
        #                     sounds_path=music_musan,
        #                     min_snr_in_db=5,
        #                     max_snr_in_db=15,
        #                     noise_transform=PolarityInversion(),
        #                     p=1,
        #                 ),
        #                 ApplyImpulseResponse(
        #                     rirs,
        #                     p=1,
        #                     leave_length_unchanged=True,
        #                     lru_cache_size=6000
        #                 ),
        #             ],p=0.5)
        #     ]
        # )
        self.augment_wav = AugmentWAV(musan_path=args.musan_path, rir_path=args.rir_path, max_frames = args.max_frames)
    def __call__(self, path, label):
        audio=loadWAV(path, max_frames=self.max_frames, evalmode=False, num_eval=10) 
        # feat=self.instance_augment(samples=feat.astype(np.float32), sample_rate=16000)
        if random.randint(0,1) > 0:
            return audio,label
        augtype = random.randint(0,4)
        if augtype == 1:
            audio   = self.augment_wav.reverberate(audio)
        elif augtype == 2:
            audio   = self.augment_wav.additive_noise('music',audio)
        elif augtype == 3:
            audio   = self.augment_wav.additive_noise('speech',audio)
        elif augtype == 4:
            audio   = self.augment_wav.additive_noise('noise',audio)
        return audio,label

    
def get_train_transform(args):
    return TrainTransform(args)
    def train_transform(path, label):
        feat=loadWAV(path, max_frames=args.max_frames, evalmode=False, num_eval=10) 
        return feat,label 
    return train_transform
def get_eval_transform(args):
    def eval_transform(path, label):
        feat=loadWAV(path, max_frames=args.eval_frames, evalmode=True, num_eval=10) 
        return feat,label 
        
class SpeakerDataset(object):
    def __init__(self, path_to_file_train, transform, args=None):
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
        count_spk = {}
        for i in self.speaker:
            if i not in count_spk:
                count_spk[i] = 0
            count_spk[i] = count_spk[i]+1
            if(count_spk[i]) > args.max_seg_per_spk:
                count_spk[i] = args.max_seg_per_spk
        total = sum([i for i in count_spk.values()])
        self.t=total
        print("Total sample ",self.t)
    def __len__(self):return len(self.speaker)

    def __getitem__(self, idx):
        path=self.paths[idx]
        label = self.speaker[idx] 
        regions=self.regions[idx]
        label=(label,regions)
        path,label = self.transform(path, label)
        # print(path.shape)
        return path, label 
    def get_batch_sampler(self, batch_size, samples_per_class,super_classes_per_batch,max_seg_per_spk ):
        speaker = np.array([
            i for i in self.speaker
        ]).reshape(-1,)
        region = np.array([
            i for i in self.regions 
        ]).reshape(-1,)
        label = np.stack([speaker, region], axis=1)
        batches_per_super_tuple=1
        sp=HierarchicalSampler(
                label,
                batch_size,
                samples_per_class=samples_per_class,
                batches_per_super_tuple=batches_per_super_tuple,
                super_classes_per_batch=super_classes_per_batch,
                inner_label=0,
                outer_label=1,
                max_seg_per_spk=max_seg_per_spk
        )
        idx=len(sp)
        while idx < self.t//batch_size:
            idx=idx + len(sp)
            batches_per_super_tuple=batches_per_super_tuple + 1
        sp=HierarchicalSampler(
                    label,
                    batch_size,
                    samples_per_class=samples_per_class,
                    batches_per_super_tuple=batches_per_super_tuple-1,
                    super_classes_per_batch=super_classes_per_batch,
                    inner_label=0,
                    outer_label=1,
                    max_seg_per_spk=max_seg_per_spk
            )
        print(f"batches_per_super_tuple = {batches_per_super_tuple-1}")
        return sp 

    def get_label(self):
        return self.speaker
    def state_dict(self):
        return {
            "path_to_file_train":self.path_to_file_train,
            "dict_speaker":self.dict_speaker,
            "type_sampler":"hirechical_sampler"
        }


class test_dataset_loader(object):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames;
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)