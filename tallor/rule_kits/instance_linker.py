from posixpath import normcase
from tqdm import tqdm

def word_norm(word):
    return ''.join([c for c in word if c!=' '])

class InstanceLinker:

    def __init__(self, opt, dataset, multi_words_threshold=0.75):
        
        self.opt = opt        
        self.multi_words = dict()
        with open('./data/'+dataset+f'/AutoPhrase_multi-words.txt') as dict_f:
            for line in dict_f.readlines():
                line = line.strip().split('\t')
                score, term = float(line[0]), line[1]
                if score>multi_words_threshold:
                    self.multi_words[word_norm(term)] = score

    def pipeline(self, instance_dict):

        print('Start linking the instances.')
        for data_idx, data_dict in tqdm(instance_dict.items(), ncols=100):
            matched_instance = []
            for span, instance in data_dict.items():
                entity = instance.entity.lower()
                if  word_norm(entity) in self.multi_words:

                    matched_instance.append((span[1]-span[0], instance))
                

            matched_instance = sorted(matched_instance, key=lambda x: x[0], reverse=True)

            for entry in matched_instance:

                instance = entry[1]

                if instance.parent == None:
                    instance.parent = True
                    start_idx, end_idx = instance.span

                    for span in enumerate_spans(start_idx, end_idx, self.opt.max_span_width):

                        if span!=instance.span:
                            
                            child = data_dict[span]

                            if child.parent == None:
                                child.parent = instance
                                instance.children.append(child)




def enumerate_spans(start_idx, end_idx, max_span_width, min_span_width=1):

    for start_index in range(start_idx, end_idx+1):
        last_end_index = min(start_index + max_span_width, end_idx+1)
        first_end_index = min(start_index + min_span_width - 1, end_idx+1)
        for end_index in range(first_end_index, last_end_index):
            start = start_index
            end = end_index
            yield (start, end)             