from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,AirAbsorption, SomeOf, Gain,RoomSimulator,SevenBandParametricEQ,HighPassFilter,LowPassFilter,AddGaussianSNR
import numpy as np
from audiomentations import OneOf,Mp3Compression,TimeMask
import soundfile as sf
import os
from tqdm import tqdm


augment = SomeOf((1,3),[
    OneOf(
        [
              AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
              AddGaussianSNR(
                  min_snr_in_db=5.0,
                  max_snr_in_db=40.0,
                  p=1.0
              ),            
        ]
    ),

    TimeStretch(min_rate=0.5, max_rate=2.5, p=1),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
    AirAbsorption(p=1),
    Gain(min_gain_in_db=-40, max_gain_in_db=40,p=1),
    Mp3Compression( p=0.5),
    RoomSimulator(p=1),
    SevenBandParametricEQ(p=1),
    HighPassFilter(p=1),
    LowPassFilter(p=1),
    TimeMask(p=1)

])
# from torchaudio_augmentations import RandomApply
# from torchaudio_augmentations import Compose, HighLowPass,ComposeMany
import matplotlib.pyplot as plt
import torch
from IPython.display import Audio

class DataSet:
  def __init__(self, paths,augment):
    self.paths=paths
    self.augment=augment
  
  def __len__(self):return len(self.paths)
  def __getitem__(self, index): 
    path=self.paths[index]
    new_path = os.path.basename(path).replace(".wav",f"_{0}.wav")
    new_path=os.path.join("../data/split_ds/augment",new_path)
    if os.path.isfile(new_path):return index
    sample,fr = sf.read(path)
    sample = sample.astype(np.float32)
    if sample.shape[0]/fr < 1:
      print("ignore sample")
      return index
    # frs.append(fr)
    assert fr==16000, path
   
    new_path = os.path.basename(path).replace(".wav",f"_{0}.wav")
    new_path=os.path.join("../data/split_ds/augment",new_path)
    augmented_samples = self.augment(samples=sample, sample_rate=fr)
    sf.write(new_path, augmented_samples, fr)
    return index
import glob
import pandas as pd
# Augmentation list_dev  datasets
list_dev=glob.glob("../data/split_ds/dev_data/*.wav")
list_dev=list(list_dev)
data=pd.DataFrame({"path":list_dev})
paths=data.path
ds=DataSet(paths,augment)
dl=torch.utils.data.DataLoader(
    ds,
    batch_size=2,
    num_workers=1,
    shuffle=False
)
for i, data in enumerate(dl):
    print(len(data))
    continue
#for i in tqdm(dl,total=len(dl)):
  #continue



