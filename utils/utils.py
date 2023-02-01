from dataset.speaker_verification_data import test_dataset_loader
import time
import itertools
import torch
import os,sys,json
import random
def evaluateFromList(model, test_list, test_path, nDataLoaderThread, print_interval=100, num_eval=4, **kwargs):
    rank = 0
    model.eval()

    lines = []
    files = []
    feats = {}
    tstart = time.time()
    ## Read all lines
    with open(test_list) as f:
        lines = f.readlines()

    ## Get a list of unique file names
    files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
    setfiles = list(set(files))
    setfiles.sort()
    ## Define test data loader
    test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)
   
    sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, shuffle=False,
        num_workers=nDataLoaderThread,
        drop_last=False, sampler=sampler
    )
    
    dc=[]
    dx=[]
    ## Extract features for every image
    for idx, data in enumerate(test_loader):
        inp1 = data[0][0].cuda() #bs,10,num_audio
        dc.append(inp1)
        dx.append(data[1][0])
        if len(dc)==64//num_eval or idx==len(test_loader)-1:
          d_c=torch.stack(dc,0)# bs,5,num_audi
          nxx=d_c.size(1)
          bs=d_c.size(0)
          with torch.no_grad():
              ref_feat = model(d_c.reshape(bs * nxx,-1)).detach().cpu()
              ref_feat=ref_feat.reshape(bs,nxx,-1)# bs,-1
          for k,x in enumerate(dx):
            feats[x] = ref_feat[k]
          dx=[]
          d_c.to("cpu")
          dc=[]
        telapsed = time.time() - tstart

        if idx % print_interval == 0 and rank == 0:
            sys.stdout.write(
                "\rReading {:d} of {:d}: {:.2f} Hz, embedding size".format(idx, test_loader.__len__(), idx / telapsed)
            )

    all_scores = []
    all_labels = []
    all_trials = []


    if rank == 0:

        tstart = time.time()
        print("")


        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]
            score = torch.nn.CosineSimilarity()(ref_feat,com_feat) # 10,10 
            score = torch.mean(score).cpu().item()
            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                sys.stdout.flush()

    return (all_scores, all_labels, all_trials)
