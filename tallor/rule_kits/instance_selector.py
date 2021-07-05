from tallor.sentence_encoder import BERTSentenceEncoder
from collections import defaultdict
import torch
import math
from tqdm import tqdm
import random
import numpy as np
import logging
import sys
class InstanceSelector:

    def __init__(self, opt, timestamp, ner_label):

        self.global_sample_times = opt.global_sample_times
        self.threshold_sample_times = opt.threshold_sample_times
        self.ner_label = ner_label
        self.logger = set_logger(timestamp)
        self.temperature = opt.temperature

        encoder_name = opt.encoder
        if encoder_name == 'bert':
            pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
            lower = 'uncased' in pretrain_ckpt
            sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, lower)
        elif encoder_name == 'scibert':
            pretrain_ckpt = opt.pretrain_ckpt or 'allenai/scibert_scivocab_uncased'
            lower = 'uncased' in pretrain_ckpt
            sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, lower)
        else:
            raise NotImplementedError

        if torch.cuda.is_available():
            self.encoder = sentence_encoder.cuda()
        else:
            self.encoder = sentence_encoder

        self.threshold_dict = dict()
        self.high_precision_repr_dict = dict()

        self.candidate_set = set()
        self.candidate_repr_dict = dict()

    def score_pipeline(self, high_precision_instances, positive_list):

        if len(high_precision_instances)==0:
            
            self.init_selector(positive_list)

            return positive_list, self.candidate_set

        self.logger.info('Begin select instances...')
        self.update_threshold(high_precision_instances)
        selected_list = []
        to_select_instance = list(set(positive_list) | self.candidate_set)
        scores = defaultdict(list)
        for instance in tqdm(to_select_instance, ncols=100):

            if instance.label == self.ner_label.get_neg_id():
                continue

            if instance in self.candidate_repr_dict:
                instance_repr = self.candidate_repr_dict[instance]
            else:
                instance_repr = self.encoder.encode_instance(instance)

            score = self.final_score(high_precision_instances[instance.label], instance_repr)
            scores[instance.label].append((instance.entity, instance.ground_truth, score))

            if score >= self.threshold_dict[instance.label] * self.temperature:
                selected_list.append(instance)
                self.high_precision_repr_dict[instance] = instance_repr
            else:
                self.candidate_set.add(instance)
                self.candidate_repr_dict[instance] = instance_repr

        if len(selected_list)==0:  # if nothing selected, we add all instance
            selected_list = positive_list
            for instance in selected_list:
                self.high_precision_repr_dict[instance] = self.candidate_repr_dict[instance]
                self.candidate_set.remove(instance)

        log_info = ''
        for label, score in scores.items():

            score = sorted(score, key=lambda x: -x[-1])
            log_info+= f'Label: {label}, Score: {score}\n'

        self.logger.info(log_info)
        self.logger.info(f'We select {len(selected_list)} instances from {len(to_select_instance)} instances.')

        return selected_list, self.candidate_set

    
    def init_selector(self, positive_list):

        print('Init instance selector.')

        for instance in positive_list:

            self.high_precision_repr_dict[instance] = self.encoder.encode_instance(instance)

    
    def update_threshold(self, high_precision_instances):

        print('Update instance selector threshold.')

        for label, instance_set in high_precision_instances.items():

            sampled_instances = random.choices(list(instance_set), k=self.threshold_sample_times)

            scores = []

            for instance in sampled_instances:

                score = self.final_score(instance_set-{instance}, self.high_precision_repr_dict[instance])
                scores.append(score)

            self.threshold_dict[label] = min(scores)
        
        log_info = ''
        for label, threshold in self.threshold_dict.items():

            log_info += f'Label: {label}, threshold: {threshold}\n'

        self.logger.info(log_info)

    def global_score(self, high_pre_set, instance_repr):

        scores = []

        for i in range(self.global_sample_times):

            sampled_instances = random.choices(list(high_pre_set), k=3)
            
            sampled_repr = np.mean(np.array([self.high_precision_repr_dict[instance] for instance in sampled_instances]), axis=0)

            score = cosine(sampled_repr, instance_repr)

            scores.append(score)

        return sum(scores)/len(scores)



    def local_score(self, high_pre_set, instance_repr):

        max_score = -1

        for instance in high_pre_set:

            score = cosine(self.high_precision_repr_dict[instance], instance_repr)

            if score > max_score:

                max_score = score

        return max_score
    
    def final_score(self, high_pre_set, instance_repr):

        global_s = self.global_score(high_pre_set, instance_repr)

        #return global_s
        local_s = self.local_score(high_pre_set, instance_repr)

        if global_s < 0 or local_s < 0:

            if global_s < local_s:
                
                return global_s
            
            else:
                return local_s

        score = global_s * local_s

        return math.sqrt(score)

def cosine(a, b):

    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def set_logger(timestamp):
    ## set logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('InstanceSelector')
    
    file_handler = logging.FileHandler(f'./logging/InstanceSelector-{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger