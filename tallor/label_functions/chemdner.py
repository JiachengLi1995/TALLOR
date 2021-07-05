from tallor.label_functions.LF_template import *
from tallor.rule_kits.rule_reader import surface_reader

class CHEMDNER_SurfaceForm(SurfaceForm):
    def __init__(self, ner_label):

        dictionary = surface_reader('chemdner', number=40)
        negative_set = set()
        
        super().__init__(ner_label, dictionary, negative_set)



class CHEMDNER_Prefix(Prefix):
    def __init__(self, ner_label):

        prefix_dict = dict()
        
        neg_prefix_set = set()

        super().__init__(ner_label, prefix_dict, neg_prefix_set)

        
class CHEMDNER_Suffix(Suffix):
    def __init__(self, ner_label):

        suffix_dict = dict()

        neg_suffix_set = set()

        super().__init__(ner_label, suffix_dict, neg_suffix_set)

class CHEMDNER_InclusivePreNgram(InclusivePreNgram):

    def __init__(self, ner_label):

        inclusive_pre_dict = dict()
        
        neg_inclusive_pre_set = set()

        super().__init__(ner_label, inclusive_pre_dict, neg_inclusive_pre_set)


class CHEMDNER_InclusivePostNgram(InclusivePostNgram):

    def __init__(self, ner_label):

        inclusive_post_dict = dict()
        neg_inclusive_post_set = set()
        super().__init__(ner_label, inclusive_post_dict, neg_inclusive_post_set)


class CHEMDNER_ExclusivePreNgram(ExclusivePreNgram):

    def __init__(self, ner_label):

        exclusive_pre_dict = dict()
        
        neg_exclusive_pre_set = set()

        super().__init__(ner_label, exclusive_pre_dict, neg_exclusive_pre_set)


class CHEMDNER_ExclusivePostNgram(ExclusivePostNgram):

    def __init__(self, ner_label):

        exclusive_post_dict = dict()

        neg_exclusive_post_set = set()

        super().__init__(ner_label, exclusive_post_dict, neg_exclusive_post_set)

class CHEMDNER_PosTagRule(PosTagRule):

    def __init__(self):

        POS_set = {'NUM':'Entity'}
        neg_POS_set = {}
        super().__init__(POS_set, neg_POS_set)

class CHEMDNER_CapitalRule(CapitalRule):

    def __init__(self):

        Capitalized = {'capitalized', 'upper'}
        super().__init__(Capitalized)
        
class CHEMDNER_DependencyRule(DependencyRule):

    def __init__(self):

        Dep_dict = dict()
        neg_Dep_set = set()
        super().__init__(Dep_dict, neg_Dep_set)

class CHEMDNER_ComposedRule(ComposedRule):

    def __init__(self, ner_label):

        ExPre = CHEMDNER_ExclusivePreNgram(ner_label)
        ExPost = CHEMDNER_ExclusivePostNgram(ner_label)
        POStag = CHEMDNER_PosTagRule()
        DepRule = CHEMDNER_DependencyRule()
        composed_rule = [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]

        # 1. [(ExPre, ExPost)]
        # 2. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]
        super().__init__(ner_label, composed_rule)


