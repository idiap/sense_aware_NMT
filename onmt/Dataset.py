'''
This software is derived from the OpenNMT project at 
https://github.com/OpenNMT/OpenNMT.
Modified 2017, Idiap Research Institute, http://www.idiap.ch/ (Xiao PU)
'''

from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, srcData, tgtData, factor_1, factor_2, factor_3, factor_4, factor_5, batchSize, cuda, volatile=False):
        self.src = srcData
        self.factor_1 = factor_1 # full tensors for whole document
        self.factor_2 = factor_2
        self.factor_3 = factor_3
        self.factor_4 = factor_4
        self.factor_5 = factor_5


        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
     
        srcBatch, lengths = self._batchify(self.src[index*self.batchSize:(index+1)*self.batchSize], align_right=False, include_lengths=True)
        # print srcBatch
        # print lengths

        #XPU: add all factorBatch and weightBatch with batch size into Dictionary

        # for i in range(1,16):
        #     globals()['factor_' + str(i)], lengths = self._batchify(getattr(self, 'factor_' + str(i))[index*self.batchSize:(index+1)*self.batchSize],align_right=False, include_lengths=True)

        factor_1, lengths = self._batchify(
            self.factor_1[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)          

        factor_2, lengths = self._batchify(
            self.factor_2[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)         
        
        factor_3, lengths = self._batchify(
            self.factor_3[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True) 
        
        factor_4, lengths = self._batchify(
            self.factor_4[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True) 
        
        factor_5, lengths = self._batchify(
            self.factor_5[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)  

        #END

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None


        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch, factor_1, factor_2, factor_3, factor_4, factor_5) if tgtBatch is None \
        else zip(indices, srcBatch, factor_1, factor_2, factor_3, factor_4, factor_5, tgtBatch)
        
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        
        if tgtBatch is None:
            indices, srcBatch, factor_1, factor_2, factor_3, factor_4, factor_5 = zip(*batch)
        else:
            indices, srcBatch, factor_1, factor_2, factor_3, factor_4, factor_5, tgtBatch = zip(*batch)



        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)


        return (wrap(srcBatch), lengths), (wrap(factor_1), lengths), (wrap(factor_2), lengths), (wrap(factor_3), lengths), \
        (wrap(factor_4), lengths), (wrap(factor_5), lengths), wrap(tgtBatch), indices

    def __len__(self):
        return self.numBatches


    # def shuffle(self):
    #     data = list(zip(self.src, self.feature, self.tgt))
    #     self.src, self.feature, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
