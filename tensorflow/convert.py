import cPickle
import pdb
import os
import glob
data_Name = "cotra"
vocab_file = "./vocab_" + data_Name + ".pkl"
input_file = './text/rec_train_words.txt'

word, vocab = cPickle.load(open('./'+vocab_file))
# input_file = 'save/coco_451.txt'
output_file = './text/sents/rec_train_sents.txt'
with open(output_file, 'w')as fout:
    with open(input_file)as fin:
        for line in fin:
            #line.decode('utf-8')
            line = line.split()
            #line.pop()
            #line.pop()
            line = [int(x) for x in line]
            line = [word[x] for x in line]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)#.encode('utf-8'))

input_file = './text/rec_val_words.txt'
# input_file = 'save/coco_451.txt'
output_file = './text/sents/rec_val_sents.txt'
with open(output_file, 'w')as fout:
    with open(input_file)as fin:
        for line in fin:
            #line.decode('utf-8')
            line = line.split()
            #line.pop()
            #line.pop()
            line = [int(x) for x in line]
            line = [word[x] for x in line]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)#.encode('utf-8'))
