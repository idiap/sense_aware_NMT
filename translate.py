'''
This software is derived from the OpenNMT project at 
https://github.com/OpenNMT/OpenNMT.
Modified 2017, Idiap Research Institute, http://www.idiap.ch/ (Xiao PU)
'''

# -*- coding: utf-8 -*-
from __future__ import division

import onmt
import torch
import argparse
import math

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def addtwo(f,g):
    for line1, line2 in zip(f,g):
        yield [line1, line2]
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, fea1_Batch, fea2_Batch, fea3_Batch, fea4_Batch, fea5_Batch, tgtBatch = [], [], [], [], [], [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    for line in addone(open(opt.src)):
        if line is not None:
                   
            srcTokens = []
            fea1_Tokens = []
            fea2_Tokens = []
            fea3_Tokens = []
            fea4_Tokens = []
            fea5_Tokens = []
            for i in line.split():
                srcTokens.append(i.strip().split("|")[0])
                fea1_Tokens.append(i.strip().split("|")[1])
                fea2_Tokens.append(i.strip().split('|')[2])
                fea3_Tokens.append(i.strip().split('|')[3])
                fea4_Tokens.append(i.strip().split('|')[4])
                fea5_Tokens.append(i.strip().split('|')[5])
                

            # srcTokens = line.split()
            srcBatch += [srcTokens]
            fea1_Batch +=[fea1_Tokens]
            fea2_Batch +=[fea2_Tokens]
            fea3_Batch +=[fea3_Tokens]
            fea4_Batch +=[fea4_Tokens]
            fea5_Batch +=[fea5_Tokens]

            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break
        predBatch, predScore, goldScore = translator.translate(srcBatch, fea1_Batch, fea2_Batch, fea3_Batch, fea4_Batch, fea5_Batch, tgtBatch)
 
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

                print('')

        srcBatch, fea1_Batch, fea2_Batch, fea3_Batch, fea4_Batch, fea5_Batch, tgtBatch = [], [], [], [], [], [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()


if __name__ == "__main__":
    main()
