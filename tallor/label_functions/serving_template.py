from tallor.label_functions.LF_template import *

class TALLOR_SurfaceForm(SurfaceForm):
    def __init__(self, ner_label):

        #dictionary = dict()
        ## using bc5cdr's rules as examples, you can write your own rules here.
        dictionary = {'proteinuria': 'Disease', 'esrd': 'Disease', 'thrombosis': 'Disease', 'tremor': 'Disease', 'hepatotoxicity': 'Disease',
                    'hypertensive': 'Disease', 'thrombotic microangiopathy': 'Disease', 'thrombocytopenia': 'Disease', 'akathisia':'Disease','confusion':'Disease',
                    'nicotine': 'Chemical', 'morphine': 'Chemical', 'haloperidol': 'Chemical', 'warfarin': 'Chemical', 'clonidine': 'Chemical',
                    'creatinine': 'Chemical', 'sirolimus': 'Chemical', 'tacrolimus': 'Chemical','isoproterenol': 'Chemical', 'cyclophosphamide': 'Chemical'}
        negative_set = set()

        ## Update labels from rules:
        ner_label.get_id('') ## negative
        for label in dictionary.values():
            ner_label.get_id(label)
        
        super().__init__(ner_label, dictionary, negative_set)



class TALLOR_Prefix(Prefix):
    def __init__(self, ner_label):

        prefix_dict = dict()
        
        neg_prefix_set = set()

        super().__init__(ner_label, prefix_dict, neg_prefix_set)

        
class TALLOR_Suffix(Suffix):
    def __init__(self, ner_label):

        suffix_dict = dict()

        neg_suffix_set = set()

        super().__init__(ner_label, suffix_dict, neg_suffix_set)

class TALLOR_InclusivePreNgram(InclusivePreNgram):

    def __init__(self, ner_label):

        inclusive_pre_dict = dict()
        
        neg_inclusive_pre_set = set()

        super().__init__(ner_label, inclusive_pre_dict, neg_inclusive_pre_set)


class TALLOR_InclusivePostNgram(InclusivePostNgram):

    def __init__(self, ner_label):

        inclusive_post_dict = dict()
        neg_inclusive_post_set = set()
        super().__init__(ner_label, inclusive_post_dict, neg_inclusive_post_set)


class TALLOR_ExclusivePreNgram(ExclusivePreNgram):

    def __init__(self, ner_label):

        exclusive_pre_dict = dict()
        
        neg_exclusive_pre_set = set()

        super().__init__(ner_label, exclusive_pre_dict, neg_exclusive_pre_set)


class TALLOR_ExclusivePostNgram(ExclusivePostNgram):

    def __init__(self, ner_label):

        exclusive_post_dict = dict()

        neg_exclusive_post_set = set()

        super().__init__(ner_label, exclusive_post_dict, neg_exclusive_post_set)

class TALLOR_PosTagRule(PosTagRule):

    def __init__(self):

        POS_set = {'NUM':'Entity'}
        neg_POS_set = {}
        super().__init__(POS_set, neg_POS_set)

class TALLOR_CapitalRule(CapitalRule):

    def __init__(self):

        Capitalized = {'capitalized', 'upper'}
        super().__init__(Capitalized)
        
class TALLOR_DependencyRule(DependencyRule):

    def __init__(self):

        Dep_dict = dict()
        neg_Dep_set = set()
        super().__init__(Dep_dict, neg_Dep_set)

class TALLOR_ComposedRule(ComposedRule):

    def __init__(self, ner_label):

        ExPre = TALLOR_ExclusivePreNgram(ner_label)
        ExPost = TALLOR_ExclusivePostNgram(ner_label)
        POStag = TALLOR_PosTagRule()
        DepRule = TALLOR_DependencyRule()
        composed_rule = [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]

        # 1. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]
        super().__init__(ner_label, composed_rule)