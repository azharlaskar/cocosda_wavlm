import glob
from scipy.io import wavfile
import os
files = glob.glob('../data/musan/noise/*/*.wav')
audlen = 16000*5
audstr = 16000*3
for idx,file in enumerate(files):
    fs,aud = wavfile.read(file)
    writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
    os.makedirs(writedir)
    for st in range(0,len(aud)-audlen,audstr):
        wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])
        print(idx,file)
