
import math
import copy
import torch
import numpy as np
def cos_cdist(x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
    #Cosine Similarity
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
    cdist = x1 @ x2.T
    cdist.div_(norms1).div_(norms2)
    return cdist
class OnlineHDv1:
    def __init__(self, classes : int, features : int, dim : int = 400, batch_size=1,lr=.0003):
        #Configure for hdb, hdc, and hde classes
        # print("test")
        self.mu=0
        self.sigma=1
        self.nClasses = classes
        self.nFeatures= features
        #hypervector size
        self.dimensionality=dim
        self.learningrate=lr
        self.batch_size=batch_size
        self.base = torch.empty(self.dimensionality).uniform_(0.0, 2*math.pi)
        #encoder
        self.hde=None
        #classifier
        self.hdc=None
        # Initialize basis in gaussian distribution
        self.basis = torch.normal(0,1,size=(self.dimensionality,self.nFeatures))
        # Initialize classification hypervectors
        self.classes = torch.zeros((self.nClasses, self.dimensionality))
        # self.prevacc=0
        # self.trainfunctions=[self.train,self.train2,self.train3]
        # self.learningrate=.1
        # self.hdc = HD_classifier(self.dimensionality, self.nClasses, 0)
        # self.trainaccuracies=[]
        # self.testaccuracies=[]
        # self.medians=[]
    def __call__(self, x : torch.Tensor):
        #return predicted values
        return self.predict(x)
    def encode(self,x):
        n = x.size(0)
        bsize = min([x.size(1),1024])
        h = torch.empty(n, self.basis.shape[0], device=x.device, dtype=x.dtype)
        temp = torch.empty(bsize, self.basis.shape[0], device=x.device, dtype=x.dtype)

        # we need batches to remove memory usage
        for i in range(0, n, bsize):
            torch.matmul(x[i:i+bsize], self.basis.T, out=temp)

            # self.noise ... I haven't seen any indication that it works better 
            # if self.noise:
            torch.add(temp, self.base, out=h[i:i+bsize])#h[i:i+bsize]=temp# torch.add(temp, self.base, out=h[i:i+bsize])
            # else:
            # h[i:i+bsize]=temp
            h[i:i+bsize].cos_().mul_(temp.sin_())
        # print(h.shape)
        return h
    def train(self,h,y):
        # print("1")
        # r=torch.randperm(y.size(0))
        # y=y[r]
        # h=h[r,:]
        batch_size = self.batch_size
        n = h.size(0)
        # print(self.learningrate)
        # print(batch_size)
        for i in range(0, n, batch_size):
            h_ = h[i:i+batch_size]
            y_ = y[i:i+batch_size]
            scores = cos_cdist(h_, self.classes)#cos
            y_pred = scores.argmax(1)
            # print(y_pred)
            wrong = y_ != y_pred

            # computes alphas to update model
            # alpha1 = 1 - delta[lbl] -- the true label coefs
            # alpha2 = delta[max] - 1 -- the prediction coefs
            aranged = torch.arange(h_.size(0), device=h_.device)
            alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
            alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

            for lbl in y_.unique():#range(0,9)
                m1 = wrong & (y_ == lbl) # mask of missed true lbl
                m2 = wrong & (y_pred == lbl) # mask of wrong preds
                # print(m1,m2)
                self.classes[lbl] += self.learningrate*(alpha1[m1]*h_[m1]).sum(0)
                # print("starting1")
                # print((self.learningrate*(alpha1[m1]*h_[m1]).sum(0)).type())
                self.classes[lbl] += self.learningrate*(alpha2[m2]*h_[m2]).sum(0)
    

    def predict(self,x):
        #return predictions based on similarity of encoded inputs to classification hypervectors
        return  cos_cdist(self.encode(x), self.classes).argmax(1)
    def fit(self,traindata, trainlabels,
                   epochs 
                    ):
        # find encoded training vectors
            # compute new encoded data
        trainencoded = self.encode(traindata)
        # testencoded = self.encode(x_testtorch)
        
        # print("regenloop: " + str(i))
        # train for specified number of epochs
        # Do the train 
        for j in range(epochs):
            # do one pass of training
            # print(self.classes[:,8])
            self.train(trainencoded, trainlabels)
            # trainaccuracy= self.test(trainencoded,trainlabels)
            # testaccuracy= self.test(testencoded,y_testtorch)
            # print(trainaccuracy)
        #            --------------

            # self.classes=torch.nn.functional.normalize(self.classes)
        # self.batch_size=int(np.ceil(self.batch_size/2))
        # if self.batch_size==1:
        #     self.learningrate=self.learningrate/2
        return 

    def test(self,x_encoded, y_labels):
            yhat= cos_cdist(x_encoded, self.classes).argmax(1)
            return (yhat==y_labels).float().mean()
    def test2(self,x_encoded,y_labels):
        yhat=torch.zeros(y_labels.size(0))
        i=0
        for v in x_encoded:
            sims=torch.matmul(v,self.classes.T)
            yhat[i]=torch.argmax(sims)
            i+=1
        return (yhat==y_labels).float().mean()
