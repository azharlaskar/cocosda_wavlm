PATH_DATA= "/media/azhar/DRIVE_2/cocosda/data/"
import glob,os,sys
from tqdm import tqdm
import multiprocessing
import time
import librosa
import soundfile as sf
import numpy as np
a=[]
def read_and_split(xs):
  # print(xs)
  print(f"Process {len(xs)}")
  for path in tqdm(xs):
    data, samplerate = sf.read(path)
    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    samplerate=16000
    chunk_size = 5 * samplerate
    for index,step in enumerate(range(0,len(data),chunk_size)):
        audio = data[step:step+chunk_size]
        new_path = os.path.basename(path).replace(".wav",f"_{index}.wav")
        new_path=os.path.join("../data/split_ds/dev_data",new_path)
        if audio.shape[0] / samplerate < 1.:
          continue
        sf.write(new_path, audio, samplerate)
  print("Done process")
    # this does not work
xss=list(glob.glob(f"{PATH_DATA}/I_MSV_DEV_ENR/Dev_data/*.wav"))
processes = [multiprocessing.Process(target=read_and_split, args=(x,)) for x in np.array_split(xss, 20)]
[p.start() for p in processes]
result = [p.join() for p in processes]
print("-"*50)
print("Done dev datasets")
print("-"*50)
def read_and_split1(xs):
  # print(xs)
  print(f"Process {len(xs)}")
  for path in tqdm(xs):
    data, samplerate = sf.read(path)
    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    samplerate=16000
    chunk_size = 5 * samplerate
    for index,step in enumerate(range(0,len(data),chunk_size)):
        audio = data[step:step+chunk_size]
        new_path = os.path.basename(path).replace(".wav",f"_{index}.wav")
        new_path=os.path.join("../data/split_ds/env_data",new_path)
        if audio.shape[0] / samplerate < 1.:
          continue
        sf.write(new_path, audio, samplerate)
xss=list(glob.glob(f"{PATH_DATA}/I_MSV_DEV_ENR/Enr_data/*.wav"))
processes = [multiprocessing.Process(target=read_and_split1, args=(x,)) for x in np.array_split(xss, 20)]
[p.start() for p in processes]
result = [p.join() for p in processes]
print(result)

print("-"*50)
print("Done env datasets")
print("-"*50)


