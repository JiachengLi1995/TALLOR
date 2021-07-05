from tallor.label_functions.LF_template import *
from tallor.rule_kits.rule_reader import surface_reader

class CONLL_SurfaceForm(SurfaceForm):
    def __init__(self, ner_label):

        dictionary = surface_reader('conll2003', number=20)  ## only start from positive surface name rule
        negative_set = set()
        
        super().__init__(ner_label, dictionary, negative_set)



class CONLL_Prefix(Prefix):
    def __init__(self, ner_label):

        prefix_dict = dict()
        
        neg_prefix_set = set()

        super().__init__(ner_label, prefix_dict, neg_prefix_set)

        
class CONLL_Suffix(Suffix):
    def __init__(self, ner_label):

        suffix_dict = dict()

        neg_suffix_set = set()

        super().__init__(ner_label, suffix_dict, neg_suffix_set)

class CONLL_InclusivePreNgram(InclusivePreNgram):

    def __init__(self, ner_label):

        inclusive_pre_dict = dict()
        
        neg_inclusive_pre_set = set()

        super().__init__(ner_label, inclusive_pre_dict, neg_inclusive_pre_set)


class CONLL_InclusivePostNgram(InclusivePostNgram):

    def __init__(self, ner_label):

        inclusive_post_dict = dict()
        neg_inclusive_post_set = set()
        super().__init__(ner_label, inclusive_post_dict, neg_inclusive_post_set)


class CONLL_ExclusivePreNgram(ExclusivePreNgram):

    def __init__(self, ner_label):

        exclusive_pre_dict = dict()
        
        neg_exclusive_pre_set = set()

        super().__init__(ner_label, exclusive_pre_dict, neg_exclusive_pre_set)


class CONLL_ExclusivePostNgram(ExclusivePostNgram):

    def __init__(self, ner_label):

        exclusive_post_dict = dict()

        neg_exclusive_post_set = set()

        super().__init__(ner_label, exclusive_post_dict, neg_exclusive_post_set)

class CONLL_PosTagRule(PosTagRule):

    def __init__(self):

        POS_set = {'NUM':'Entity'}
        neg_POS_set = {}
        super().__init__(POS_set, neg_POS_set)

class CONLL_CapitalRule(CapitalRule):

    def __init__(self):

        Capitalized = {'capitalized', 'upper'}
        super().__init__(Capitalized)
        
class CONLL_DependencyRule(DependencyRule):

    def __init__(self):

        Dep_dict = dict()
        neg_Dep_set = set()
        super().__init__(Dep_dict, neg_Dep_set)

class CONLL_ComposedRule(ComposedRule):

    def __init__(self, ner_label):

        ExPre = CONLL_ExclusivePreNgram(ner_label)
        ExPost = CONLL_ExclusivePostNgram(ner_label)
        POStag = CONLL_PosTagRule()
        DepRule = CONLL_DependencyRule()
        composed_rule = [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]

        #1. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost), (DepRule, POStag)]
        #2. [(ExPre, ExPost), (DepRule, POStag)]
        #3. [(ExPre, ExPost), (ExPre, POStag), (POStag, ExPost)]
        #4. [(ExPre, ExPost)]
        super().__init__(ner_label, composed_rule)


