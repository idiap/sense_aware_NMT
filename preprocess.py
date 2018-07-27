'''
This software is derived from the OpenNMT project at 
https://github.com/OpenNMT/OpenNMT.
Modified 2017, Idiap Research Institute, http://www.idiap.ch/ (Xiao PU)
'''

# -*- coding: utf-8 -*-
import onmt

import argparse
import torch



parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-feature_vocab', help="Path to an existing feature vocabulary")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")
parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")
parser.add_argument('-feature', type=int, default=0, help='1 contains feature, 0 is no feature')



opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word.split("|")[0])

    originalSize = vocab.size()
    vocab = vocab.prune(size)

    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')

        genWordVocab = makeVocabulary(dataFile, vocabSize)
        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, featureDicts, tgtDicts):
    src, tgt = [], []   
    sizes = []
    count, ignored = 0, 0
    
    #XPU: add more numbers of factors here
    factor_1, factor_2, factor_3, factor_4, factor_5 = [], [], [], [], [] # index File for whole document

    #END

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        
        srcWords = [] # store words for one line

        feaWords_1, feaWords_2, feaWords_3, feaWords_4, feaWords_5 = [], [], [], [], [] # index File for whole document

        # #END


        for position, words in enumerate(sline.split()):
            srcWords.append(words.split("|")[0])

            #XPU: add here
            feaWords_1.append(words.split('|')[1])
            feaWords_2.append(words.split('|')[2])
            feaWords_3.append(words.split('|')[3])
            feaWords_4.append(words.split('|')[4])
            feaWords_5.append(words.split('|')[5])
            #END

        tgtWords = tline.split()
    

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.seq_length:

            src += [srcDicts.convertToIdx(srcWords,onmt.Constants.UNK_WORD)]

            factor_1 += [featureDicts.convertToIdx(feaWords_1,onmt.Constants.UNK_WORD)]
            factor_2 += [featureDicts.convertToIdx(feaWords_2,onmt.Constants.UNK_WORD)]
            factor_3 += [featureDicts.convertToIdx(feaWords_3,onmt.Constants.UNK_WORD)]
            factor_4 += [featureDicts.convertToIdx(feaWords_4,onmt.Constants.UNK_WORD)]
            factor_5 += [featureDicts.convertToIdx(feaWords_5,onmt.Constants.UNK_WORD)]
            #END
            #print factor_5

            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()


    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        # print str(len(src))
        
        #XPU: add here

        factor_1 = [factor_1[idx] for idx in perm]
        factor_2 = [factor_2[idx] for idx in perm]
        factor_3 = [factor_3[idx] for idx in perm]
        factor_4 = [factor_4[idx] for idx in perm]
        factor_5 = [factor_5[idx] for idx in perm]
        #END

        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]

    #XPU: add here
    factor_1 = [factor_1[idx] for idx in perm]
    factor_2 = [factor_2[idx] for idx in perm]
    factor_3 = [factor_3[idx] for idx in perm]
    factor_4 = [factor_4[idx] for idx in perm]
    factor_5 = [factor_5[idx] for idx in perm]
    #END

    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt, factor_1, factor_2, factor_3, factor_4, factor_5


def main():

    dicts = {}


    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,opt.src_vocab_size)
    dicts['feature'] = initVocabulary('features', opt.train_src, opt.feature_vocab, opt.src_vocab_size)
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)
    

    print('Preparing training ...')
    train = {}

    train['src'], train['tgt'], train['trainFactor_1'], train['trainFactor_2'], \
    train['trainFactor_3'], train['trainFactor_4'], train['trainFactor_5'] = makeData(opt.train_src, opt.train_tgt, dicts['src'], dicts['feature'], dicts['tgt'])


    print('Preparing validation ...')
    valid = {}

    valid['src'], valid['tgt'], valid['validFactor_1'], valid['validFactor_2'], \
    valid['validFactor_3'], valid['validFactor_4'], valid['validFactor_5'] = makeData(opt.valid_src, opt.valid_tgt,
                                    dicts['src'],dicts['feature'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')
    # saveVocabulary('feature', feature_dicts['feature'], opt.save_data + '.feature.dict')


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts, 'train': train, 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
