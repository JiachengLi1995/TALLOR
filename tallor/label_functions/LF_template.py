from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from collections import defaultdict
from tqdm import tqdm

def word_normalize(word):

    return ''.join((e.lower() for e in word if e.isalnum() or e==' '))


def make_lf(function_name, function, resources):
    return LabelingFunction(
        name=function_name,
        f=function,
        resources=resources,
    )

class SurfaceForm:

    def __init__(self, ner_label, dictionary, negative_set):

        self.ner_label = ner_label
        self.pos_dict = dictionary
        self.neg_set = negative_set

    def get_label_functions(self):

        self.label_functions = self.generate_label_functions()
        
        return self.label_functions

    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()


    def match(self, x, surface, label):

        sentence = x.sentence
        span = x.span
        entity = x.entity.lower()
        parsed_tokens = x.parsed_tokens
        entity_lemma = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])
        
        if surface == entity or surface == entity_lemma:

            return label
        
        else:

            return -1 #ABSTAIN

    def pre_match(self, x):
        
        span = x.span
        entity = x.entity.lower()
        parsed_tokens = x.parsed_tokens
        entity_lemma = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])

        return entity in self.pos_dict or entity_lemma in self.pos_dict

    def generate_label_functions(self):

        self.neg_set = set([word.lower() for word in self.neg_set])

        @labeling_function(resources=dict(pos_dict=self.pos_dict, neg_set=self.neg_set, ner_label=self.ner_label))
        def lf_in_dict(x, pos_dict, neg_set, ner_label):

            sentence = x.sentence
            span = x.span
            entity = x.entity.lower()
            parsed_tokens = x.parsed_tokens
            entity_lemma = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])
            pos_set = [w.pos_ for w in parsed_tokens[span[0]:span[1]+1]]

            if entity in neg_set or entity_lemma in neg_set:

                return ner_label.get_neg_id()

            label = None

            if entity in pos_dict:
                
                label = pos_dict[entity]

            elif entity_lemma in pos_dict:

                label = pos_dict[entity_lemma]
                
            else:

                return -1 #ABSTAIN


            if label and ner_label.is_exist(label):
                    
                return ner_label.get_id(label)

            else:

                return  ner_label.get_neg_id()

        return [lf_in_dict]


    def update_label_functions(self, rules, label):

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:
                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances):

        rule_candidates = defaultdict(list)

        for instance in instances:
            span = instance.span
            parsed_tokens = instance.parsed_tokens
            
            entity_lemma = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])
            
            if entity_lemma not in self.pos_dict and entity_lemma not in self.neg_set:
                rule_candidates[entity_lemma].append(instance)

        return rule_candidates


            

class Prefix:

    def __init__(self, ner_label, prefix_dict, neg_prefix_set):

        self.ner_label = ner_label
        self.pos_dict = prefix_dict
        self.neg_set = neg_prefix_set      
        
    def get_label_functions(self):

        self.label_functions = []
        self.generate_label_functions()

        return self.label_functions

    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, prefix, label):

        sentence = x.sentence
        parsed_tokens = x.parsed_tokens
        span = x.span
        entity = x.entity.lower()
        entity_lemmas = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])

        # entity_root = x.entity_root.text.lower()
        # entity_root_lemma = x.entity_root.lemma_.lower()

        if entity.startswith(prefix) or entity_lemmas.startswith(prefix):

            return label
        
        else:
            return -1 #ABSTAIN

    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False


    def generate_label_functions(self):

        for prefix, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'prefix_{prefix}', 
                        function=self.match, 
                        resources=dict(prefix=prefix, label=label_id))

            else:

                lf = make_lf(function_name=f'prefix_{prefix}', 
                        function=self.match, 
                        resources=dict(prefix=prefix, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)

        for prefix in self.neg_set:

            lf = make_lf(function_name=f'neg_prefix_{prefix}', 
                        function=self.match, 
                        resources=dict(prefix=prefix, label=self.ner_label.get_neg_id()))
                
            
            self.label_functions.append(lf)
    
    def update_label_functions(self, rules, label):

        # self.pos_dict = dict()
        # self.neg_set = set()

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances, len_list = [4, 5, 6]):

        rule_candidates = defaultdict(list)

        for instance in instances:
            
            rule_list = self.generate_single_rule(instance, len_list)
            for rule in rule_list:
                rule_candidates[rule].append(instance)

        return rule_candidates

    def generate_single_rule(self, instance, len_list = [4, 5, 6]):

        parsed_tokens = instance.parsed_tokens
        span = instance.span
        #entity_root_lemma = instance.entity_root.lemma_.lower()
        entity_lemmas = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])
        rule_list = []
        for length in len_list:
            if len(entity_lemmas)>=length and entity_lemmas[:length] not in self.pos_dict and entity_lemmas[:length] not in self.neg_set:
                rule_list.append(entity_lemmas[:length])

        return rule_list
                

class Suffix:

    def __init__(self, ner_label, suffix_dict, neg_suffix_set):

        self.ner_label = ner_label
        self.pos_dict = suffix_dict
        self.neg_set = neg_suffix_set
        
    def get_label_functions(self):

        self.label_functions = []
        self.generate_label_functions()
        
        return self.label_functions

    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, suffix, label):

        sentence = x.sentence
        parsed_tokens = x.parsed_tokens
        span = x.span
        entity = x.entity.lower()
        entity_lemmas = ' '.join([w.lemma_ for w in parsed_tokens[span[0]:span[1]+1]])
        # entity_root = x.entity_root.text.lower()
        # entity_root_lemma = x.entity_root.lemma_.lower()

        if entity.endswith(suffix) or entity_lemmas.endswith(suffix):

            return label
        
        else:
            return -1 #ABSTAIN

    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False

    def generate_label_functions(self):

        for suffix, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'suffix_{suffix}', 
                        function=self.match, 
                        resources=dict(suffix=suffix, label=label_id))

            else:

                lf = make_lf(function_name=f'suffix_{suffix}', 
                        function=self.match, 
                        resources=dict(suffix=suffix, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)

        for suffix in self.neg_set:

            lf = make_lf(function_name=f'neg_suffix_{suffix}', 
                        function=self.match, 
                        resources=dict(suffix=suffix, label=self.ner_label.get_neg_id()))
                
            
            self.label_functions.append(lf)

    def update_label_functions(self, rules, label):

        # self.pos_dict = dict()
        # self.neg_set = set()

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances, len_list = [4, 5, 6]):

        rule_candidates = defaultdict(list)

        for instance in instances:
            
            rule_list = self.generate_single_rule(instance, len_list)
            for rule in rule_list:
                rule_candidates[rule].append(instance)

        return rule_candidates

    def generate_single_rule(self, instance, len_list = [4, 5, 6]):

        #entity_root_lemma = instance.entity_root.lemma_.lower()
        parsed_tokens = instance.parsed_tokens
        span = instance.span
        entity_lemmas = ' '.join([w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]])
        rule_list = []
        for length in len_list:

            if len(entity_lemmas)>=length and entity_lemmas[-length:] not in self.pos_dict and entity_lemmas[-length:] not in self.neg_set:
                rule_list.append(entity_lemmas[-length:])

        return rule_list


class InclusivePreNgram:

    def __init__(self, ner_label, inclusive_pre_dict, neg_inclusive_pre_set):

        self.ner_label = ner_label
        self.pos_dict = inclusive_pre_dict
        self.neg_set = neg_inclusive_pre_set
        
    def get_label_functions(self):
        
        self.label_functions = []
        self.generate_label_functions()
        
        return self.label_functions
    
    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, inclusive_pre, label):

        sentence = x.sentence
        parsed_tokens = x.parsed_tokens
        span = x.span
        words = sentence[span[0]:span[1]+1]
        lemmas = [w.lemma_ for w in parsed_tokens[span[0]:span[1]+1]]
        inclusive_pre_splitted = inclusive_pre.split(' ')
        pre_len = len(inclusive_pre_splitted)
        
        if len(words)<=pre_len:
            return -1 #ABSTAIN
        else:
            word = ' '.join(words[:pre_len]).lower()
            lemma = ' '.join(lemmas[:pre_len]).lower()

            if word==inclusive_pre or lemma==inclusive_pre:

                return label
            
            else:
                return -1 #ABSTAIN
    
    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False

    def generate_label_functions(self):

        for inclusive_pre, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'inclusive_pre_{inclusive_pre}', 
                        function=self.match, 
                        resources=dict(inclusive_pre=inclusive_pre, label=label_id))

            else:

                lf = make_lf(function_name=f'inclusive_pre_{inclusive_pre}', 
                        function=self.match, 
                        resources=dict(inclusive_pre=inclusive_pre, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)

        for inclusive_pre in self.neg_set:

            lf = make_lf(function_name=f'neg_inclusive_pre_{inclusive_pre}', 
                        function=self.match, 
                        resources=dict(inclusive_pre=inclusive_pre, label=self.ner_label.get_neg_id()))
                
            
            self.label_functions.append(lf)

    def update_label_functions(self, rules, label):

        # self.pos_dict = dict()
        # self.neg_set = set()

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances, ngram_list = [1,2,3]):   

        rule_candidates = defaultdict(list)

        for instance in instances:
            
            rule_list = self.generate_single_rule(instance, ngram_list)
            for rule in rule_list:
                rule_candidates[rule].append(instance)  

        return rule_candidates

    def generate_single_rule(self, instance, ngram_list = [1,2,3]):

        parsed_tokens = instance.parsed_tokens
        span = instance.span
        lemmas = [w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]]
        rule_list = []
        
        for ngram in ngram_list:

            if len(lemmas)>=ngram:

                inclusive_pre_lemma = ' '.join(lemmas[:ngram])
                if inclusive_pre_lemma not in self.pos_dict and inclusive_pre_lemma not in self.neg_set:
                    rule_list.append(inclusive_pre_lemma)

        return rule_list

        
class InclusivePostNgram:

    def __init__(self, ner_label, inclusive_post_dict, neg_inclusive_post_set):

        self.ner_label = ner_label
        self.pos_dict = inclusive_post_dict
        self.neg_set = neg_inclusive_post_set
        
    def get_label_functions(self):

        self.label_functions = []
        self.generate_label_functions()

        return self.label_functions
    
    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, inclusive_post, label):

        sentence = x.sentence
        parsed_tokens = x.parsed_tokens
        span = x.span
        words = sentence[span[0]:span[1]+1]
        lemmas = [w.lemma_ for w in parsed_tokens[span[0]:span[1]+1]]
        inclusive_post_splitted = inclusive_post.split(' ')
        post_len = len(inclusive_post_splitted)
        
        if len(words)<=post_len:
            return -1 #ABSTAIN
        else:
            word = ' '.join(words[-post_len:]).lower()
            lemma = ' '.join(lemmas[-post_len:]).lower()

            if word==inclusive_post or lemma==inclusive_post:

                return label
            
            else:
                return -1 #ABSTAIN

    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False

    def generate_label_functions(self):

        for inclusive_post, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'inclusive_post_{inclusive_post}', 
                        function=self.match, 
                        resources=dict(inclusive_post=inclusive_post, label=label_id))

            else:

                lf = make_lf(function_name=f'inclusive_post_{inclusive_post}', 
                        function=self.match, 
                        resources=dict(inclusive_post=inclusive_post, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)

        for inclusive_post in self.neg_set:

            lf = make_lf(function_name=f'neg_inclusive_post_{inclusive_post}', 
                        function=self.match, 
                        resources=dict(inclusive_post=inclusive_post, label=self.ner_label.get_neg_id()))
                
            
            self.label_functions.append(lf)

    def update_label_functions(self, rules, label):

        # self.pos_dict = dict()
        # self.neg_set = set()

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances, ngram_list = [1,2,3]):  

        rule_candidates = defaultdict(list)

        for instance in instances:
            
            rule_list = self.generate_single_rule(instance, ngram_list)
            for rule in rule_list:
                rule_candidates[rule].append(instance)

        return rule_candidates

    def generate_single_rule(self, instance, ngram_list = [1,2,3]):

        parsed_tokens = instance.parsed_tokens
        span = instance.span
        lemmas = [w.lemma_.lower() for w in parsed_tokens[span[0]:span[1]+1]]
        rule_list = []

        for ngram in ngram_list:
            if len(lemmas)>=ngram:

                inclusive_post_lemma = ' '.join(lemmas[-ngram:])
                if inclusive_post_lemma not in self.pos_dict and inclusive_post_lemma not in self.neg_set:
                    rule_list.append(inclusive_post_lemma)

        return rule_list


class ExclusivePreNgram:

    def __init__(self, ner_label, exclusive_pre_dict, neg_exclusive_pre_set):

        self.ner_label = ner_label
        self.pos_dict = exclusive_pre_dict
        self.neg_set = neg_exclusive_pre_set
        
    def get_label_functions(self):

        self.label_functions = []
        self.generate_label_functions()
        
        return self.label_functions

    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, exclusive_pre, label):

        sentence = x.sentence
        parsed_tokens = x.parsed_tokens
        span = x.span
        
        exclusive_pre_splitted = exclusive_pre.split()
        pre_len = len(exclusive_pre_splitted)
        if span[0]<pre_len-1:
            return -1 #ABSTAIN
        
        if span[0]==pre_len-1:

            words = ['[START]']+sentence[span[0]-pre_len+1:span[0]]
            lemmas = ['[START]']+[w.lemma_ for w in parsed_tokens[span[0]-pre_len+1:span[0]]]

        else:
            words = sentence[span[0]-pre_len:span[0]]
            lemmas = [w.lemma_ for w in parsed_tokens[span[0]-pre_len:span[0]]]
        
        word = ' '.join(words).lower()
        lemma = ' '.join(lemmas).lower()

        if word==exclusive_pre or lemma==exclusive_pre:

            return label
        
        else:
            return -1 #ABSTAIN

    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False

    def generate_label_functions(self):

        for exclusive_pre, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'exclusive_pre_{exclusive_pre}', 
                        function=self.match, 
                        resources=dict(exclusive_pre=exclusive_pre, label=label_id))

            else:

                lf = make_lf(function_name=f'exclusive_pre_{exclusive_pre}', 
                        function=self.match, 
                        resources=dict(exclusive_pre=exclusive_pre, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)

        for exclusive_pre in self.neg_set:

            lf = make_lf(function_name=f'neg_exclusive_pre_{exclusive_pre}', 
                        function=self.match, 
                        resources=dict(exclusive_pre=exclusive_pre, label=self.ner_label.get_neg_id()))
                
            
            self.label_functions.append(lf)

    def update_label_functions(self, rules, label):

        # self.pos_dict = dict()
        # self.neg_set = set()

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances, ngram_list = [1,2,3]): 

        rule_candidates = defaultdict(list)

        for instance in instances:

            rule_list = self.generate_single_rule(instance, ngram_list)

            for rule in rule_list:
                rule_candidates[rule].append(instance)

        return rule_candidates

    def generate_single_rule(self, instance, ngram_list = [1,2,3]):

        parsed_tokens = instance.parsed_tokens
        span = instance.span
        rule_list = []

        for ngram in ngram_list:

            if span[0]>=ngram-1:
                if span[0]==ngram-1:
                    lemmas = ['[START]']+[w.lemma_.lower() for w in parsed_tokens[span[0]-ngram:span[0]]]
                else:
                    lemmas = [w.lemma_.lower() for w in parsed_tokens[span[0]-ngram:span[0]]]
                exclusive_pre_lemma = ' '.join(lemmas)
                if exclusive_pre_lemma not in self.pos_dict and exclusive_pre_lemma not in self.neg_set:
                    
                    rule_list.append(exclusive_pre_lemma)

        return rule_list



class ExclusivePostNgram:

    def __init__(self, ner_label, exclusive_post_dict, neg_exclusive_post_set):

        self.ner_label = ner_label
        self.pos_dict = exclusive_post_dict
        self.neg_set = neg_exclusive_post_set
        
    def get_label_functions(self):

        self.label_functions = []
        self.generate_label_functions()
        
        return self.label_functions

    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, exclusive_post, label):

        sentence = x.sentence
        parsed_tokens = x.parsed_tokens
        span = x.span
        
        exclusive_post_splitted = exclusive_post.split()
        post_len = len(exclusive_post_splitted)
        if len(sentence)-span[1]-1<post_len-1:
            return -1 #ABSTAIN

        if len(sentence)-span[1]-1==post_len-1:
            words = sentence[span[1]+1:span[1]+post_len+1]+['[END]']
            lemmas = [w.lemma_ for w in parsed_tokens[span[1]+1:span[1]+post_len+1]]+['[END]']
        else:
            words = sentence[span[1]+1:span[1]+post_len+1]
            lemmas = [w.lemma_ for w in parsed_tokens[span[1]+1:span[1]+post_len+1]]
        
        word = ' '.join(words).lower()
        lemma = ' '.join(lemmas).lower()

        if word==exclusive_post or lemma==exclusive_post:

            return label
        
        else:
            return -1 #ABSTAIN

    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False

    def generate_label_functions(self):

        for exclusive_post, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'exclusive_post_{exclusive_post}', 
                        function=self.match, 
                        resources=dict(exclusive_post=exclusive_post, label=label_id))

            else:

                lf = make_lf(function_name=f'exclusive_post_{exclusive_post}', 
                        function=self.match, 
                        resources=dict(exclusive_post=exclusive_post, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)

        for exclusive_post in self.neg_set:

            lf = make_lf(function_name=f'neg_exclusive_post_{exclusive_post}', 
                        function=self.match, 
                        resources=dict(exclusive_post=exclusive_post, label=self.ner_label.get_neg_id()))
                
            
            self.label_functions.append(lf)

    def update_label_functions(self, rules, label):

        # self.pos_dict = dict()
        # self.neg_set = set()

        for rule in rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances, ngram_list = [1,2,3]): 

        rule_candidates = defaultdict(list)

        for instance in instances:

            rule_list = self.generate_single_rule(instance, ngram_list) 

            for rule in rule_list:
                           
                rule_candidates[rule].append(instance)

        return rule_candidates

    def generate_single_rule(self, instance, ngram_list = [1,2,3]):

        parsed_tokens = instance.parsed_tokens
        span = instance.span
        rule_list = []
        for ngram in ngram_list:

            if len(parsed_tokens)-span[1]-1>=ngram-1:
                
                if len(parsed_tokens)-span[1]-1==ngram-1:
                    lemmas = [w.lemma_.lower() for w in parsed_tokens[span[1]+1:span[1]+ngram+1]] + ['[END]']
                else:
                    lemmas = [w.lemma_.lower() for w in parsed_tokens[span[1]+1:span[1]+ngram+1]]
                    
                exclusive_post_lemma = ' '.join(lemmas)
                if exclusive_post_lemma not in self.pos_dict and exclusive_post_lemma not in self.neg_set:
                    rule_list.append(exclusive_post_lemma)

        return rule_list

class PosTagRule:

    def __init__(self, POS_dict, neg_POS_set):

        self.pos_dict = POS_dict
        self.neg_set = neg_POS_set


    def match(self, x, pos_rule, label):

        parsed_tokens = x.parsed_tokens
        span = x.span

        pos_tag = ' '.join([w.pos_ for w in parsed_tokens[span[0]:span[1]+1]])

        if pos_tag==pos_rule:
            return label
        else:
            return -1

    def pre_match(self, x):

        parsed_tokens = x.parsed_tokens
        span = x.span

        pos_tag = ' '.join([w.pos_ for w in parsed_tokens[span[0]:span[1]+1]])
        
        if pos_tag in self.pos_dict:

            return True
        
        return False

    def generate_single_rule(self, x):

        parsed_tokens = x.parsed_tokens
        span = x.span

        pos_tag = ' '.join([w.pos_ for w in parsed_tokens[span[0]:span[1]+1]])

        return [pos_tag]

class DependencyRule:
    def __init__(self, Dep_dict, neg_Dep_set):

        self.pos_dict = Dep_dict
        self.neg_set = neg_Dep_set


    def match(self, x, Dep_rule, label):


        parsed_tokens = x.parsed_tokens
        span = x.span
        rule_splited = Dep_rule.split('||')
        
        one_hop, two_hops = self.get_dep_features(parsed_tokens, span)

        
        if len(rule_splited)==1:

            if rule_splited[0] == one_hop:
                return label
            else:
                return -1
        
        elif len(rule_splited)==2:

            if rule_splited[0] == one_hop and rule_splited[1] in two_hops:
                return label
            else:
                return -1
        
        else:
            return -1

    def pre_match(self, x):

        parsed_tokens = x.parsed_tokens
        span = x.span

        for rule, label in self.pos_dict.items():

            if self.match(x, rule, label)==label:

                return True

        return False

    def generate_single_rule(self, x):

        parsed_tokens = x.parsed_tokens
        span = x.span
        one_hop, two_hops = self.get_dep_features(parsed_tokens, span)

        dep_rules = [one_hop]
        for two_hop in two_hops:

            dep_rules.append(one_hop+'||'+two_hop)

        return dep_rules

    def get_dep_features(self, parsed_tokens, span):

        ancestor = self.common_ancestor(parsed_tokens, span)
        one_hop = ancestor.head
        one_hop_text = one_hop.lemma_.lower()
        two_hop_text = set()
    
        for child in one_hop.children:
            if child.i<span[0] or child.i>span[1]:
                two_hop_text.add(child.lemma_.lower())
        
        return one_hop_text, two_hop_text

    def common_ancestor(self, parsed_tokens, span):
        ## this ancestor must in the span
        head_list = []
        for idx in range(span[0], span[1]+1):
            head_idx = parsed_tokens[idx].head.i
            if head_idx<span[0] or head_idx>span[1]:
                head_list.append(idx)

        if len(head_list)==1:
            return parsed_tokens[head_list[0]]
        else:
            return parsed_tokens[span[1]]

## only used for instance negative filter
class CapitalRule:

    def __init__(self, Capitalized):

        self.capitalized = Capitalized
        

    def pre_match(self, x):

        parsed_tokens = x.parsed_tokens
        span = x.span
        patient = 0
        if 'capitalized' in self.capitalized:

            flag = True

            for w in parsed_tokens[span[0]:span[1]+1]:

                if not w.is_title:
                    if patient==0:
                        if w.is_stop or w.is_digit or w.is_upper:
                            patient+=1
                            continue
                    flag = False
                    break

            if flag:
                return True
        
        if 'upper' in self.capitalized:

            flag = True

            for w in parsed_tokens[span[0]:span[1]+1]:

                if not w.is_stop and not w.is_upper:
                    flag = False
                    break
            if flag:
                return True

        return False
    
class ComposedRule:

    def __init__(self, ner_label, composed_rule):

        self.ner_label = ner_label
        
        self.all_rules = composed_rule
        self.pos_dict = dict()
        self.neg_set = set()
       
    def get_label_functions(self):

        self.label_functions = []
        self.generate_label_functions()
        
        return self.label_functions

    def clear_rule(self):

        self.pos_dict = dict()
        self.neg_set = set()

    def match(self, x, composed_rule, label):

        left_rule, right_rule, composed_idx = composed_rule
        rule_pattern = self.all_rules[composed_idx]
        left_rule_pattern, right_rule_pattern = rule_pattern

        left_label = left_rule_pattern.match(x, left_rule, label)
        right_label = right_rule_pattern.match(x, right_rule, label)

        if left_label==label and right_label==label:

            return label
        
        else:

            return -1 #ABSTAIN
    
    def pre_match(self, x):
        
        for key, value in self.pos_dict.items():

            if self.match(x, key, value)!=-1:

                return True
        
        return False

    def generate_label_functions(self):

        for composed_rule, label in self.pos_dict.items():

            if self.ner_label.is_exist(label):

                label_id = self.ner_label.get_id(label)

                lf = make_lf(function_name=f'composed_{composed_rule}', 
                        function=self.match, 
                        resources=dict(composed_rule=composed_rule, label=label_id))

            else:

                lf = make_lf(function_name=f'composed_{composed_rule}', 
                        function=self.match, 
                        resources=dict(composed_rule=composed_rule, label=self.ner_label.get_neg_id()))

            self.label_functions.append(lf)


    def update_label_functions(self, composed_rules, label):

        for rule in composed_rules:

            if label == self.ner_label.get_neg_id():

                self.neg_set.add(rule)

            else:

                self.pos_dict[rule] = self.ner_label.get_label(label)

    def generate_new_rules(self, instances): 

        rule_candidates = defaultdict(list)

        for i, (left_rule_pattern, right_rule_pattern) in tqdm(enumerate(self.all_rules), ncols=100, total=len(self.all_rules)):
            
            for instance in instances:

                left_rule_list = left_rule_pattern.generate_single_rule(instance)
                right_rule_list = right_rule_pattern.generate_single_rule(instance)

                for left_rule in left_rule_list:
                    for right_rule in right_rule_list:

                        rule_candidates[(left_rule, right_rule, i)].append(instance)

        return rule_candidates 