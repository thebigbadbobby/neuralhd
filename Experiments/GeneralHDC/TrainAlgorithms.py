from Module import *

class OnlineHDv2(TrainProcess):
    def __init__(self,epochs,batchsize,learningrate):
        self.algorithm=TrainRepeater(ClassicVersion2(batchsize,learningrate),epochs)
    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
class NeuralHDv2(TrainProcess):
    def __init__(self,epochs,regenloops,batchsize,learningrate, dropoutrate):
        if dropoutrate==0:
            trainblock=TrainBlock(
            [Normalize(),OnlineHDv2(epochs,batchsize,learningrate)]
            )
        else:
            trainblock=TrainBlock(
            [ResetLowVarianceDims(.1), Normalize(),OnlineHDv2(epochs,batchsize,learningrate)]
            )
        self.algorithm=TrainRepeater(trainblock,regenloops) # repeat regenloops times

    def apply(self,model,train,label,xtr):
        self.algorithm.apply(model,train,label,xtr)
