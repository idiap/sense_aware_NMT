'''
This software is derived from the OpenNMT project at 
https://github.com/OpenNMT/OpenNMT.
Modified 2017, Idiap Research Institute, http://www.idiap.ch/ (Xiao PU)
'''

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.Attention_Xiao import Attention_Xiao
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import numpy as np


class Encoder(nn.Module):

    def __init__(self, opt, dicts, feature_dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size * 2
        self.label_index = feature_dicts.size() - 1

        super(Encoder, self).__init__()

        self.weight_attn = Attention_Xiao(opt.word_vec_size)
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size, padding_idx=onmt.Constants.PAD)

        #initial_feature_embedding:
        self.feature_lut = nn.Embedding(feature_dicts.size(), opt.word_vec_size, padding_idx=onmt.Constants.PAD)
        #print self.feature_lut.shape()
        #END

        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = np.loadtxt(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(torch.from_numpy(pretrained))
            

    def load_pretrained_feature_vectors(self, opt):
        if opt.pre_feature_vecs_enc is not None:

            feature_initial = np.loadtxt(opt.pre_feature_vecs_enc)
            self.feature_lut.weight.data.copy_(torch.from_numpy(feature_initial))


    def forward(self, input, factor_1, factor_2, factor_3, factor_4, factor_5, hidden=None):
        
        if isinstance(input, tuple):
         
            lengths = input[1].data.view(-1).tolist() # lengths data is wrapped inside a Variable
            emb_word = self.word_lut(input[0])



            padMask_1 = factor_1[0].data.eq(self.label_index) # batch * sentL   
            padMask_2 = factor_2[0].data.eq(self.label_index)       
            padMask_3 = factor_3[0].data.eq(self.label_index)
            padMask_4 = factor_4[0].data.eq(self.label_index)
            padMask_5 = factor_5[0].data.eq(self.label_index)
           

            emb_factor_1 = self.feature_lut(factor_1[0]) # batch * sentL * dim
            emb_factor_2 = self.feature_lut(factor_2[0])
            emb_factor_3 = self.feature_lut(factor_3[0])
            emb_factor_4 = self.feature_lut(factor_4[0])
            emb_factor_5 = self.feature_lut(factor_5[0])

            # XPU: Context_Vector
            # if opt.cuda:
            sum_emb_word = torch.sum(emb_word,1)[:,0,:] # batch X dim  Context_Vector (CPU warn mistake)
            # else:
            #sum_emb_word = torch.sum(emb_word,1) # works for CPU
            #print mean_emb_word.size()
            #print mean_emb_word.size()

            emb_total = [] # batch * length * dim

            for i in range(0, emb_word.size(1)):

                tempMask1 = padMask_1[:,i].unsqueeze(1) #batch_vector
                tempMask2 = padMask_2[:,i].unsqueeze(1)
                tempMask3 = padMask_3[:,i].unsqueeze(1)
                tempMask4 = padMask_4[:,i].unsqueeze(1)
                tempMask5 = padMask_5[:,i].unsqueeze(1)

                # index_1 = factor_1[0][:,i].unsqueeze(1)
                # index_2 = factor_2[0][:,i].unsqueeze(1)
                # index_3 = factor_3[0][:,i].unsqueeze(1)
                # index_4 = factor_4[0][:,i].unsqueeze(1)
                # index_5 = factor_5[0][:,i].unsqueeze(1)

                # print torch.cat([factor_1[0][:,i].unsqueeze(1), factor_2[0][:,i].unsqueeze(1), factor_3[0][:,i].unsqueeze(1), \
                #     factor_4[0][:,i].unsqueeze(1), factor_5[0][:,i].unsqueeze(1)], 1)

                temp_mean = (sum_emb_word - emb_word[:,i,:]) / float(emb_word.size(1) - 1) 

                temp_factor1 = emb_factor_1[:,i,:].unsqueeze(1)   # batch * dim
                temp_factor2 = emb_factor_2[:,i,:].unsqueeze(1) 
                temp_factor3 = emb_factor_3[:,i,:].unsqueeze(1) 
                temp_factor4 = emb_factor_4[:,i,:].unsqueeze(1) 
                temp_factor5 = emb_factor_5[:,i,:].unsqueeze(1) 

                temp_factorDis = torch.cat([temp_factor1, temp_factor2, temp_factor3, temp_factor4, temp_factor5], 1) # batch * 5feature * dim
                temp_masks = torch.cat([tempMask1, tempMask2, tempMask3, tempMask4, tempMask5], 1)  # batch * 5Feature
                #print temp_masks

                #print temp_factorDis

                self.weight_attn.applyMask(temp_masks)
                weightProduct, weightDistribution = self.weight_attn(temp_mean, temp_factorDis)

                emb_total.append(weightProduct) # batch x dim

            emb_total = torch.stack(emb_total).transpose(0,1)
            emb_total = torch.cat([emb_word, emb_total], 2)
            # print emb_total.size()
            emb = pack(emb_total, lengths)

        # else:

        #     emb_word = self.word_lut(input)
        #     emb_feature = self.feature_lut(feature)
        #     emb = torch.cat([emb_word, emb_feature],2)
           
        outputs, hidden_t = self.rnn(emb, hidden) 

        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        # print outputs.size()
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)
            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(1,0))
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input): # input is the zip(batch) without index: index, sourceBatch, factorArrayBatch, weightArrayBatch, tgtBatch
        
        src = input[0]
        #XPU: add feature input
        factor_1 = input[1]
        factor_2 = input[2]
        factor_3 = input[3]
        factor_4 = input[4]
        factor_5 = input[5]


        #END
        tgt = input[6][:-1]  # exclude last target from inputs
        
        #tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, factor_1, factor_2, factor_3, factor_4, factor_5)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)

        return out
