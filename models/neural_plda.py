class NeuralPlda(nn.Module):
    def __init__(self, nc):
        super(NeuralPlda, self).__init__()
        self.centering_and_LDA = nn.Linear(nc.xvector_dim, nc.layer1_LDA_dim)  # Centering, wccn
        self.centering_and_wccn_plda = nn.Linear(nc.layer1_LDA_dim, nc.layer2_PLDA_spkfactor_dim)
        self.P_sqrt = nn.Parameter(torch.rand(nc.layer2_PLDA_spkfactor_dim, requires_grad=True))
        self.Q = nn.Parameter(torch.rand(nc.layer2_PLDA_spkfactor_dim, requires_grad=True))
        self.threshold = {}
        for beta in nc.beta:
            self.threshold[beta] = nn.Parameter(0*torch.rand(1, requires_grad=True))
            self.register_parameter("Th{}".format(int(beta)), self.threshold[beta])
        self.threshold_Xent = nn.Parameter(0*torch.rand(1, requires_grad=True))
        self.threshold_Xent.requires_grad = False
        self.alpha = torch.tensor(nc.alpha).to(nc.device)
        self.beta = nc.beta
        self.dropout = nn.Dropout(p=0.5)
        self.lossfn = nc.loss


    def extract_plda_embeddings(self, x):
        x = self.centering_and_LDA(x)
        x = F.normalize(x)
        x = self.centering_and_wccn_plda(x)
        return x

    def forward_from_plda_embeddings(self,x1,x2):
        P = self.P_sqrt * self.P_sqrt
        Q = self.Q
        S = (x1 * Q * x1).sum(dim=1) + (x2 * Q * x2).sum(dim=1) + 2 * (x1 * P * x2).sum(dim=1)
        return S
    
    def forward(self, x1, x2):
        x1 = self.extract_plda_embeddings(x1)
        x2 = self.extract_plda_embeddings(x2)
        S = self.forward_from_plda_embeddings(x1,x2)
        return S

    def softcdet(self, output, target):
        sigmoid = nn.Sigmoid()
        losses = [((sigmoid(self.alpha * (self.threshold[beta] - output)) * target).sum() / (target.sum()) + beta * (sigmoid(self.alpha * (output - self.threshold[beta])) * (1 - target)).sum() / ((1 - target).sum())) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def crossentropy(self, output, target):
        sigmoid = nn.Sigmoid()
        loss = F.binary_cross_entropy(sigmoid(output - self.threshold_Xent), target)
        return loss
    
    def loss(self, output, target):
        if self.lossfn == 'SoftCdet':
            return self.softcdet(output, target)
        elif self.lossfn == 'crossentropy':
            return self.crossentropy(output, target)

    def cdet(self, output, target):
        losses = [((output < self.threshold[beta]).float() * target).sum() / (target.sum()) + beta * ((output > self.threshold[beta]).float() * (1 - target)).sum() / ((1 - target).sum()) for beta in self.beta]
        loss = sum(losses)/len(losses)
        return loss
    
    def minc(self, output, target, update_thresholds=False, showplots=False):
        scores_target, _ = torch.sort(output[target>0.5])
        scores_nontarget, _ = torch.sort(-output[target<0.5])
        scores_nontarget = -scores_nontarget
        pmiss_arr = [arr2val(torch.where(scores_target < i)[0], -1) for i in scores_target]
        pmiss = torch.tensor(pmiss_arr).float() / (target.cpu().sum())
        pfa_arr = [arr2val(torch.where(scores_nontarget >= i)[0], -1) for i in scores_target]
        pfa = torch.tensor(pfa_arr).float() / ((1-target.cpu()).sum())
        cdet_arr, minc_dict, minc_threshold = {}, {}, {}
        for beta in self.beta:
            cdet_arr[beta] = pmiss + beta*pfa
            minc_dict[beta], thidx = torch.min(cdet_arr[beta], 0)
            minc_threshold[beta] = scores_target[thidx]
            if update_thresholds:
                self.state_dict()["Th{}".format(int(beta))].data.copy_(minc_threshold[beta])
        mincs = list(minc_dict.values())
        minc_avg = sum(mincs)/len(mincs)
        if showplots:
            plt.figure()
            minsc = output.min()
            maxsc = output.max()
            plt.hist(np.asarray(scores_nontarget), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.hist(np.asarray(scores_target), bins=np.linspace(minsc,maxsc,50), alpha=0.5, normed=True)
            plt.plot(scores_target, pmiss)
            plt.plot(scores_target, pfa)
            plt.plot(scores_target, cdet_arr[99])
            plt.plot(scores_target, cdet_arr[199])
            plt.show()
        return minc_avg, minc_threshold