from tallor.label_functions.LF_template import *
from tallor.rule_kits.rule_reader import surface_reader

class BC5CDR_SurfaceForm(SurfaceForm):
    def __init__(self, ner_label):

        dictionary = surface_reader('bc5cdr', number=20)
        negative_set = set()
        
        super().__init__(ner_label, dictionary, negative_set)



class BC5CDR_Prefix(Prefix):
    def __init__(self, ner_label):

        prefix_dict = dict()
        
        neg_prefix_set = set()

        super().__init__(ner_label, prefix_dict, neg_prefix_set)

        
class BC5CDR_Suffix(Suffix):
    def __init__(self, ner_label):

        suffix_dict = dict()

        neg_suffix_set = set()

        super().__init__(ner_label, suffix_dict, neg_suffix_set)

class BC5CDR_InclusivePreNgram(InclusivePreNgram):

    def __init__(self, ner_label):

        inclusive_pre_dict = dict()
        
        neg_inclusive_pre_set = set()

        super().__init__(ner_label, inclusive_pre_dict, neg_inclusive_pre_set)


class BC5CDR_InclusivePostNgram(InclusivePostNgram):

    def __init__(self, ner_label):

        inclusive_post_dict = dict()
        neg_inclusive_post_set = set()
        super().__init__(ner_label, inclusive_post_dict, neg_inclusive_post_set)


class BC5CDR_ExclusivePreNgram(ExclusivePreNgram):

    def __init__(self, ner_label):

        exclusive_pre_dict = dict()
        
        neg_exclusive_pre_set = set()

        super().__init__(ner_label, exclusive_pre_dict, neg_exclusive_pre_set)


class BC5CDR_ExclusivePostNgram(ExclusivePostNgram):

    def __init__(self, ner_label):

        exclusive_post_dict = dict()

        neg_exclusive_post_set = set()

        super().__init__(ner_label, exclusive_post_dict, neg_exclusive_post_set)

class BC5CDR_PosTagRule(PosTagRule):

    def __init__(self):

        POS_set = {'NUM':'Entity'}
        neg_POS_set = {}
        super().__init__(POS_set, neg_POS_set)

class BC5CDR_CapitalRule(CapitalRule):

    def __init__(self):

        Capitalized = {'capitalized', 'upper'}
        super().__init__(Capitalized)
        
class BC5CDR_DependencyRule(DependencyRule):

    def __init__(self):

        Dep_dict = dict()
        neg_Dep_set = set()
        super().__init__(Dep_dict, neg_Dep_set)

class BC5CDR_ComposedRule(ComposedRule):

    def __init__(self, ner_label):

        ExPre = BC5CDR_ExclusivePreNgram(ner_label)
        ExPost = BC5CDR_ExclusivePostNgram(ner_label)
        POStag = BC5CDR_PosTagRule()
        DepRule = BC5CDR_DependencyRule()
        composed_rule = [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]

        # 1. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]
        super().__init__(ner_label, composed_rule)


