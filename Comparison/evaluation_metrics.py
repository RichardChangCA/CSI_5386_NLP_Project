import tensorflow as tf
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
from tqdm import tqdm
import glob
import datetime
import gc
import pickle
from PIL import Image

# BLEU: Bilingual Evaluation Understudy
# evaluating a generated sentence to a referene sentence
# A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.
# paper: https://www.aclweb.org/anthology/P02-1040.pdf
# n-grams
# Andrew Ng deeplearning.ai https://www.youtube.com/watch?v=DejHQYAGb7Q&t=2s
# more packages:
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from cider import Cider

class evaluation():
    def __init__(self,model_name,encoder_name,dataset_name):
        self.model_name=model_name
        self.encoder_name=encoder_name
        self.dataset_name=dataset_name
        result_file_name_base = "./prediction_results/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name
        self.real_file = result_file_name_base+"_real.txt"
        self.predict_file = result_file_name_base+"_predict.txt"

        self.references = []
        self.candidates = []
        f_real = open(self.real_file,'r')
        f_predict = open(self.predict_file,'r')
        for real_caption, predict_caption in zip(f_real,f_predict):
            self.references.append([real_caption.split(' ')[1:-2]])
            self.candidates.append(predict_caption.split(' ')[1:-2])

        result_file_name = "./evaluation_results/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name
        self.f = open(result_file_name,'w+')
        self.f.write("model_name: "+self.model_name +'\n')
        self.f.write("encoder_name: "+self.encoder_name +'\n')
        self.f.write("dataset_name: "+self.dataset_name +'\n\n')
    
    def blue_score_call(self):
        gram_1_blue_score = corpus_bleu(self.references,self.candidates,weights=(1, 0, 0, 0))
        gram_2_blue_score = corpus_bleu(self.references,self.candidates,weights=(0, 1, 0, 0))
        gram_3_blue_score = corpus_bleu(self.references,self.candidates,weights=(0, 0, 1, 0))
        gram_4_blue_score = corpus_bleu(self.references,self.candidates,weights=(0, 0, 0, 1))
        comulative_blue_score = corpus_bleu(self.references,self.candidates,weights=(0.4, 0.3, 0.2, 0.1))

        self.f.write("1-gram blue score: " + str(gram_1_blue_score) + '\n')
        self.f.write("2-gram blue score: " + str(gram_2_blue_score) + '\n')
        self.f.write("3-gram blue score: " + str(gram_3_blue_score) + '\n')
        self.f.write("4-gram blue score: " + str(gram_4_blue_score) + '\n')
        self.f.write("comulative blue score with weights (4,3,2,1): " + str(comulative_blue_score) + '\n\n')

    def CIDEr_call(self):
        cider_instance = Cider()
        f_real = open(self.real_file,'r')
        f_predict = open(self.predict_file,'r')
        references = []
        candidates = []
        for real_caption, predict_caption in zip(f_real,f_predict):
            references.append([real_caption])
            candidates.append(predict_caption)
        cider_score = cider_instance.compute_score(candidates,references)
        self.f.write("CIDEr score: " + str(cider_score) + '\n\n')

    def METEOR_call(self):
        meteor_score_value = []
        f_real = open(self.real_file,'r')
        f_predict = open(self.predict_file,'r')
        for real_caption, predict_caption in zip(f_real,f_predict):
            meteor_score_value.append(single_meteor_score(real_caption,predict_caption))
        self.f.write("METEOR score: " + str(np.mean(meteor_score_value)) + '\n\n')

dataset_name = 'flickr8k'
model_name = 'GRU_with_Bahdanau_Attention'
encoder_name = 'InceptionV3'
evaluation_instance = evaluation(model_name,encoder_name,dataset_name)
evaluation_instance.blue_score_call()
evaluation_instance.CIDEr_call()
evaluation_instance.METEOR_call()
print(dir())