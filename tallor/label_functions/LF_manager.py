from .conll2003 import *
from .bc5cdr import *
from .chemdner import *
from .serving_template import *
class LFManager:

    def __init__(self, ner_label, dataset, rule_types='composed', mode='training'):
        
        self.rule_templates = []
        self.pre_match_templates = []

        if mode != 'training':
            self.rule_templates.append(TALLOR_SurfaceForm(ner_label))
            self.rule_templates.append(TALLOR_ComposedRule(ner_label))
            self.pre_match_templates += self.rule_templates
            self.pre_match_templates += [TALLOR_CapitalRule(), TALLOR_PosTagRule()]
        
        else:

            if dataset == 'conll2003':

                if rule_types=='composed':
                    self.rule_templates.append(CONLL_SurfaceForm(ner_label))
                    self.rule_templates.append(CONLL_ComposedRule(ner_label))

                self.pre_match_templates += self.rule_templates
                self.pre_match_templates += [CONLL_CapitalRule(), CONLL_PosTagRule()]

            elif dataset == 'bc5cdr':

                if rule_types=='composed':
                    self.rule_templates.append(BC5CDR_SurfaceForm(ner_label))
                    self.rule_templates.append(BC5CDR_ComposedRule(ner_label))

                self.pre_match_templates += self.rule_templates
                self.pre_match_templates += [BC5CDR_CapitalRule(), BC5CDR_PosTagRule()]

            elif dataset == 'chemdner':

                if rule_types=='composed':
                    self.rule_templates.append(CHEMDNER_SurfaceForm(ner_label))
                    self.rule_templates.append(CHEMDNER_ComposedRule(ner_label))

                self.pre_match_templates += self.rule_templates
                self.pre_match_templates += [CHEMDNER_CapitalRule(), CHEMDNER_PosTagRule()]

            else:
                print(f'Please complete rule template for dataset {dataset}.')
                raise NotImplementedError


    def get_all_functions(self):

        label_functions = []

        for rule_template in self.rule_templates:

            label_functions += rule_template.get_label_functions()

        return label_functions

    def generate_all_new_rules(self, instances, threshold):

        new_rules = []

        for rule_template in self.rule_templates:
            new_rule = rule_template.generate_new_rules(instances)
            new_rule_filted = dict()
            for rule_name, v in new_rule.items():
                if len(v)>threshold:
                    new_rule_filted[rule_name] = v
            new_rules.append(new_rule_filted)

        return new_rules

            

    def update_all_functions(self, selected_rules):

        for i, rule_template in enumerate(self.rule_templates):

            rule_dict = selected_rules[i]

            for label, rules in rule_dict.items():

                rule_template.update_label_functions(rules, label)


    def get_all_rules(self):

        label_rules = []

        for rule_template in self.rule_templates:

                label_rules.append(rule_template.pos_dict)
                label_rules.append(rule_template.neg_set)
        return label_rules

    def pre_match(self, x):

        for rule_template in self.pre_match_templates:

            if rule_template.pre_match(x):
                
                return True
                
        return False

    def clear_all_rules(self):

        for rule_template in self.rule_templates:

            rule_template.clear_rule()
    
    @staticmethod
    def set_ner_label(ner_label):
        TALLOR_SurfaceForm(ner_label)
        TALLOR_ComposedRule(ner_label)