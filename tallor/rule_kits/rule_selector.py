from collections import defaultdict
import numpy as np
import math
from tqdm import tqdm
import time

class RuleSelector:

    def __init__(self, opt, ner_label, logger):
        
        self.logger = logger
        self.rule_topk = LinearTopk(opt.rule_topk, 4*opt.rule_topk, 1)
        self.ner_label = ner_label
        self.opt = opt

    def pipeline(self, pos_instances, neg_instances, new_rules):

        self.logger.info(f'We have {len(pos_instances)} positive instances, {len(neg_instances)} negative instances.')
        training_dict = defaultdict(list)
        training_dict[self.ner_label.get_neg_id()] = neg_instances

        for instance in pos_instances:
            training_dict[instance.label].append(instance)

            
        selected_rules = self.simple_select(training_dict, new_rules)

        log_rules = 'Updated rules:\n'
        for i, rules in enumerate(selected_rules):
            for label, rules_list in rules.items():
                log_rules+=f'Template: {i},Label:{label}, Rules: {rules_list}\n'
                rules[label] = [line[0] for line in rules_list]
        self.logger.info(log_rules)
        return selected_rules
    
    def simple_select(self, training_dict, new_rules):

        repr_dict = dict()
        for k, v in training_dict.items():
            repr_dict[k] = set(v)

        selected_rules = []
        scores = defaultdict(list)
        for i, new_rule in enumerate(new_rules):
            selected_rules.append(defaultdict(list))
            for rule_name, instances in tqdm(new_rule.items(), ncols=100):
                instance_set = set(instances)
                s = [0]*len(repr_dict.keys())
                c = [0]*len(repr_dict.keys())
                for label, label_set in repr_dict.items():
                    Fi = len(instance_set & label_set)+1e-10
                    Ni = len(instance_set)+1e-10
                    if label != self.ner_label.get_neg_id():
                        s[label] = Fi/Ni 
                        c[label] = math.log2(Fi)
                    else:
                        s[label] = Fi/Ni 
                        c[label] = math.log2(Fi)
                pred = np.argmax(np.array(s))

                threshold = 0
                if isinstance(rule_name, tuple): ## composed rule
                    rule_id = pred
                    threshold=0.9
                else:          ## surface name
                    rule_id = pred
                    threshold = 0.9

                if pred != self.ner_label.get_neg_id():
                    if s[pred]>=threshold:
                        scores[rule_id].append((i, rule_name, s[pred], c[pred], s[pred] * c[pred]))
                else:
                    if i!=0:
                        scores[rule_id] = []#.append((i, rule_name, s[pred], c[pred], (s[pred], c[pred])))
        topk = self.rule_topk.get_topk()
        for rule_id, rule_list in scores.items():
            label = rule_id
            if label==self.ner_label.get_neg_id():
                selected = []#list(sorted(rule_list, key=lambda x: x[-1], reverse=True))[:self.rule_topk.get_topk()]
            else:
                selected = list(sorted(rule_list, key=lambda x: x[-1], reverse=True))[:topk]

            for rule in selected:
                selected_rules[rule[0]][label].append((rule[1], rule[2], rule[3]))

        return selected_rules
        

def cosine(a, b):

    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


class LinearTopk:
    def __init__(self, Min, Max, step):
        
        self.Min = Min
        self.Max = Max
        self.step = step
        self.i = 0

    def get_topk(self):

        if self.Min+self.step * self.i>self.Max:
            topk = int(self.Max)
        
        else:
            topk = int(self.Min+self.step * self.i)

        self.i+=1

        return topk