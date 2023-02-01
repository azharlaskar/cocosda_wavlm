from pytorch_lightning import LightningModule
# from audiossl.models.atst import ATST
from models.rawnet import MainModelRawnet
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset.speaker_verification_data import SpeakerDataset, test_dataset_loader, get_train_transform, get_eval_transform
from dataset.sampler import HierarchicalSampler
import time
import torch
import warnings
warnings.simplefilter("ignore")
from callbacks.callbacks import ScoreCallback
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
from utils.utils import evaluateFromList
from utils.tuneThreshold import *
from collections import OrderedDict
#from models.tdnn import ECAPA_TDNN_SMALL
#from models import tdnn
config_path = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from commons.loss_inter_aam import AAMINTER
wavlm_extractor = torch.hub.load('s3prl/s3prl',"wavlm_large")
from models.tdnn import ECAPA_TDNN_SMALL
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        # print(x.size())
        # x=x.squeeze(1)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
class SpeakerLightingModule(LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        print(kwargs)
        self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path,extractor=wavlm_extractor)
        checkpoint="./wavlm_large_finetune.pth"
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        print(self.model.load_state_dict(state_dict['model'], strict=False))
        # print(self.model.load_state_dict(torch.load("/content/sources/final_source/model.pt")['model']))
        import math 
        phi=args.margin*180/math.pi
        loss_function = losses.ArcFaceLoss(
            kwargs['nClasses'], kwargs['nOut'], margin=phi, scale=kwargs['scale'],distance=CosineSimilarity()
        )
        print(f"ArcFaceloss with scale={args.scale}\nPhi={phi}") 
        self.loss_function=AAMINTER(
            kwargs['nOut'],kwargs['nClasses'],
            margin=kwargs['margin'], scale=kwargs['scale'],
            top_k=10,margin_negative=0.1
        )
        # self.load_state_dict(torch.load("/content/drive/MyDrive/dataset/imsv/epoch=17-VEER=1.955-mindcf=0.136.ckpt")['state_dict'])
        self.save_hyperparameters()
        # self.loss_function = LossFunction(
        #     kwargs['nOut'], kwargs['nClasses'], margin=kwargs['margin'], scale=kwargs['scale']
        # )
        # self.loss_function2 = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5,)
    def forward(self, x):
        outputs = self.model(x) 
        return outputs
    def training_step(self, batch, batch_idx):
        inputs, label = batch
        inputs=inputs.float()
        speaker_label, region_label = label
        outputs = self.model(inputs[:,0,:]) 
        inputs=inputs.float()

        loss = self.loss_function(outputs, speaker_label) # + loss region

        tqdm_dict={}
        if isinstance(loss,(tuple,list)):
            acc=loss[1]
            tqdm_dict = {"acc":acc}
            self.log('train_acc', acc,prog_bar=True,)
            loss=loss[0]
        # loss =loss
        self.log('train_loss', loss)
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
            })
        return output
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, 1,eta_min=1e-5)
        print("init {} optimizer with learning rate {}".format("Adam", self.hparams.learning_rate))
        return optimizer
    def evaluate(self):
        all_scores, all_labels, all_trials=evaluateFromList(self.model, self.hparams.test_list, "", self.hparams.nDataLoaderThread,eval_frames=self.hparams.eval_frames)
        result = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, self.hparams.dcf_p_target, self.hparams.dcf_c_miss, self.hparams.dcf_c_fa)
        self.log(
            'VEER',result[1],prog_bar=True, logger=True
        )
        self.log(
            
            'mindcf',mindcf,prog_bar=True, logger=True
        )
    def train_dataloader(self):
        train_dataset = SpeakerDataset(
            self.hparams.train_list,
            get_train_transform(self.hparams),
            self.hparams
        )

        sampler_train =train_dataset.get_batch_sampler(
            self.hparams.batch_size,
            self.hparams.nPerSpeaker,
            self.hparams.nLanguage,
            max_seg_per_spk=self.hparams.max_seg_per_spk
        )
        print(len(sampler_train))
        train_loader =torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.hparams.nDataLoaderThread, batch_sampler=sampler_train) 
        return train_loader
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATSTModel")
        parser.add_argument("--arch",type=str,default="small")
        parser.add_argument("--learning_rate", default=0.0005, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")
        parser.add_argument('--ema', default=0.99, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            """)
        parser.add_argument('--warmup_steps',default=1300,type=int)
        parser.add_argument('--max_steps',default=39010,type=int)
        parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')
        parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
        parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
        parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
        parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
        parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
        parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
        parser.add_argument("--max_seg_per_spk", type=int, default=512, help="Maximum utt per speaker in one epochs")
        ## Training details
        parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
        parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
        parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

        ## Optimizer
        parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
        parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
        parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
        parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
        parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

        ## Loss functions
        parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
        parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
        parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
        parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
        parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
        parser.add_argument('--nLanguage',       type=int,   default=3,   help='Number of language in batch')
        parser.add_argument('--nClasses',       type=int,   default=15000,   help='Number of speakers in the softmax layer, only for softmax-based losses')
        parser.add_argument('--temperature',       type=float,   default=1,   help='t for margin-softmax-based losses')
        parser.add_argument('--combine_strategy',       type=str,   default="protopycal",   help='metric learning [protopycal|multi_loss|triplet]')

        ## Evaluation parameters
        parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
        parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
        parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

        ## Load and save
        parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
        parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

        ## Training and test data
        parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
        parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
        parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
        parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

        ## Model definition
        parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
        parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
        parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
        parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
        parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
        parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3') 
        return parent_parser

from argparse import ArgumentParser
parser = ArgumentParser("ATST")
import pytorch_lightning as pl
parser.add_argument('--nproc', type=int,  default=2)
parser = SpeakerLightingModule.add_model_specific_args(parser)
args = parser.parse_args()
print(vars(args))
model =SpeakerLightingModule(**vars(args))
# dataset=model.train_dataloader()
# for i in dataset:
#   print(i[0].size(),i[1][0].size())
# print(next(iter(dataset)))

trainer = pl.Trainer( max_epochs=20,log_every_n_steps =200,accelerator='cpu',check_val_every_n_epoch =1,callbacks=[ScoreCallback(),pl.callbacks.ModelCheckpoint(args.save_path,filename ="{epoch}-{VEER:.3f}-{mindcf:.3f}",save_last =True,mode ='min',monitor ="VEER",save_on_train_epoch_end =True)])
trainer.fit(model=model.to(device="cpu"))
