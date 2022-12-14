
import math
import copy
import torch
import numpy as np
from matplotlib import pyplot as plt
def cos_cdist(x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
    #Cosine Similarity
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
    cdist = x1 @ x2.T
    cdist.div_(norms1).div_(norms2)
    return cdist
class NeuralHDv2:
    def __init__(self, classes : int, features : int, dim : int = 400, batch_size=1,lr=.0003, multiencoder=True):
        #Configure for hdb, hdc, and hde classes
        # print("test")
        self.mu=0
        self.sigma=1
        self.nClasses = classes
        self.nFeatures= features
        self.multiencoder = multiencoder
        #hypervector size
        self.dimensionality=dim
        self.learningrate=lr
        self.batch_size=batch_size
        self.base = torch.empty(self.dimensionality).uniform_(0.0, 2*math.pi)

        #dimension metric
        self.dimmetric=torch.zeros(self.dimensionality)
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
        if self.multiencoder:
            for i in range(0, n, bsize):
                torch.matmul(x[i:i+bsize], self.basis.T, out=temp)

                # self.noise ... I haven't seen any indication that it works better 
                # if self.noise:
                torch.add(temp, self.base, out=h[i:i+bsize])#h[i:i+bsize]=temp# torch.add(temp, self.base, out=h[i:i+bsize])
                # else:
                # h[i:i+bsize]=temp
                h[i:i+bsize].cos_().mul_(temp.sin_())
        else:
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
    def train3(self,h,y):
        # def fit(self, data, label, param = None):
        # print("3")
        assert self.dimensionality == h.size(1)
        #if self.first_fit:
        #    sys.stderr.write("Fitting with configuration: %s \n" % str([(k,param[k]) for k in self.options]))

        # Actual fitting

        # handling dropout

        # fit
        r = torch.randperm(y.size(0))
        y=y[r]
        h=h[r,:]
        correct = 0
        count = 0
        for i in range(0,y.size(0),self.batch_size):
            sample = h[i:i+self.batch_size] 
            answers = y[i:i+self.batch_size]
            #maxVal = -1
            #guess = -1
            #for m in range(self.nClasses):
            #    val = kernel(self.classes[m], sample)
            #    if val > maxVal:
            #        maxVal = val
            #        guess = m
            vals = cos_cdist(sample, self.classes)
            # print(vals)
            guesses = vals.argmax(1)
            # print(guesses)
            for j in range(0,answers.size(0)):
                if guesses[j] != answers[j]:
                    # print(answers[j])
                    self.classes[guesses[j]]-=self.learningrate*h[i+j]*(1-vals[0,guesses[j]])
                    self.classes[answers[j]]+=self.learningrate*h[i+j]*(1-vals[0,answers[j]])
                    guessmeter=(sample[j]*self.classes[guesses[j]])
                    answermeter=(sample[j]*self.classes[answers[j]])
                    self.dimmetric+=guessmeter-answermeter
                    # acc=self.test2(h[r][:100],y)
                    # if acc<=self.prevacc:
                    #     self.classes[guess]+=self.learningrate*h[i]
                    #     self.classes[answer]-=self.learningrate*h[i]
                    # else:
                    #     self.prevacc=acc
                else:
                    correct += 1
                count += 1
        return correct / count

    def predict(self,x):
        #return predictions based on similarity of encoded inputs to classification hypervectors
        return  cos_cdist(self.encode(x), self.classes).argmax(1)
    def fit(self,traindata, trainlabels,
                   epochs,
                   regenloops,  # list of effective dimensions to reach 
                   fractionToDrop # drop/regen rate 
                    ):
        # find encoded training vectors

        # calculate amount of dropped dimensions based on percent and original dimension
        amountDrop = int(fractionToDrop * self.dimensionality)#self.param.D?
        # print("Updating times:", regenloops)

        for i in range(regenloops+1): # For each eDs to reach, will checkpoints
            # compute new encoded data
            trainencoded = self.encode(traindata)
            # testencoded = self.encode(x_testtorch)
            
            # print("regenloop: " + str(i))
            # train for specified number of epochs
            # Do the train 
            for j in range(epochs):
                # do one pass of training
                # print(self.classes[:,8])
                self.train3(trainencoded, trainlabels)
                # trainaccuracy= self.test(trainencoded,trainlabels)
                # testaccuracy= self.test(testencoded,y_testtorch)
                # print(trainaccuracy)

            
                # print(self.prevacc)
            #if its the last regeneration training, stop before doing another dimension drop; stop if 100% accuracy
            if i==regenloops:
                return #self.hdc,self.hde - unnecessary now that hdc and hde are within a class
            # print("regen" +str(i))
            #do the dimension drop and regeneration
            normed_classes = torch.nn.functional.normalize(self.classes)
            #calculate variances for each dimension
            # torch.
            # var = torch.var(normed_classes, 0) 
            # assert len(var) == self.dimensionality
            # rank each entry in variances from smallest to largest
            # order = torch.argsort(self.dimmetric)
            # breakeven=torch.argmin(abs(self.dimmetric[order]))
            trainencoded = self.encode(traindata)
            totalclearance=torch.zeros(self.dimensionality)
            for index in range(len(self.classes)):
                classmetrics=torch.Tensor([(trainencoded[trainlabels==index]*class_).sum(0).tolist() for class_ in self.classes])
                clearance=classmetrics[index]-classmetrics.mean(0)
                totalclearance+=clearance
            order = torch.argsort(totalclearance)
            # breakeven=torch.argmin(abs(totalclearance))
            #drop amountDrop bases
            toDrop = order[:amountDrop]#amountdrop
            plt.plot(range(len(order)),totalclearance[order])#self.dimmetric[order]
            plt.show()
            #            ----------------
            #attempted reverse drop
            # if amountDrop<0:
            #     toDrop = order[-amountDrop:]
            #            ----------------
            #Update basis
            #For each dimension designated to be dropped
            for i in toDrop:
                #generate a new ith vector in the basis
                self.basis[i] = torch.normal(self.mu,self.sigma, size=(self.nFeatures,))
            #Update Classes
            #            --------------
            #This code was left out. Maybe useful?
            for i in toDrop:
                self.classes[:,i] = torch.zeros(self.nClasses)
            #            --------------

            self.classes=torch.nn.functional.normalize(self.classes)
            # self.batch_size=int(np.ceil(self.batch_size/2))
            # if self.batch_size==1:
            #     self.learningrate=self.learningrate/2
        return "error","error"

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