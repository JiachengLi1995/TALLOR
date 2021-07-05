import torch
import torch.utils.data as data
import os
import random
import json
from tallor.utils import DataPoint, list_to_dict
from copy import deepcopy
import spacy
from spacy.tokens import Doc

class Parser:

    def __init__(self):

        self.parser = spacy.load('en_core_web_sm')
        self.parser.remove_pipe('ner')
        self.parser.tokenizer = self.custom_tokenizer

    def custom_tokenizer(self, text):
        tokens = text.split(' ')
        return Doc(self.parser.vocab, tokens)

    def parse(self, sentence):

        return self.parser(sentence)

class MissingDict(dict):
    """
    If key isn't there, returns default value. Like defaultdict, but it doesn't store the missing
    keys that were queried.
    """
    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val

def format_label_fields(ner):
    
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start, span_end), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    return ner_dict


class DataSet(data.Dataset):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, root, filename, encoder, batch_size, ner_label, is_train, opt, mode='training'):
        self.batch_size = batch_size
        self.max_span_width = opt.max_span_width
        self.ner_label = ner_label
        self.encoder = encoder
        ## spacy
        self.parser = Parser()  ## use customized parser to ensure that we have the same tokens

        self.label_data = []
        self.unlabel_data = []

        if mode == 'training':

            labeled_ratio = opt.labeled_ratio
            path = os.path.join(root, filename + ".json")

            data = []
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data.append(json.loads(line))
            
            print(f'Begin processing {filename} dataset...')
            
            processed_data = self.preprocess(data)

            data = []
            for line in processed_data:
                sentence, spans, ner_labels = line
                data.append(DataPoint(
                            sentence=sentence,
                            spans=spans,
                            ner_labels=ner_labels,
                            parsed_tokens=self.parser.parse(' '.join(sentence)),
                            label_num = self.ner_label.get_num()
                        ))

            labeled_num = 0
            unlabeled_num = 0
            if not is_train or labeled_ratio==1:
                self.training_data = data
                labeled_num = len(self.training_data)
                unlabeled_num = 0
            else:
                index = list(range(len(data)))
                labeled_index = index[:int(labeled_ratio*len(index))]
                unlabel_index = index[int(labeled_ratio*len(index)):]

                self.label_data = [data[i] for i in labeled_index]

                for i in unlabel_index:

                    data[i].unlabel_reset() ## set all label mask to 0
                    self.unlabel_data.append(data[i])
                    
                self.training_data = deepcopy(self.label_data)

                labeled_num = len(labeled_index)
                unlabeled_num = len(unlabel_index)
                
            print(f'Done. {filename} dataset has {len(data)} instances. \n Among them, we use {labeled_num} instances as labeled data, {unlabeled_num} instances as unlabeled data')
        else:
            data = self.read_and_process_unlabel_set(root, filename)

            self.unlabel_data = data
            self.training_data = []
            print(f'We get {len(self.unlabel_data)} sentences.')

    def read_and_process_unlabel_set(self, root, filename):

        path = os.path.join(root, filename)

        data = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line)['sentence'])

        processed_data = self.preprocess(data, mode='serving')

        data = []
        for line in processed_data:
            sentence, spans = line
            data_point = DataPoint(
                        sentence=sentence,
                        spans=spans,
                        ner_labels=[-1]*len(spans),
                        parsed_tokens=self.parser.parse(' '.join(sentence)),
                        label_num = self.ner_label.get_num()
                    )
            data_point.unlabel_reset()
            data.append(data_point)

        return data

    def preprocess(self, data, mode='training'):
        """ Preprocess the data and convert to ids. """
        processed = []
        if mode == 'training':
            for line in data:
                for sentence, ner in zip(line["sentences"], line["ner"]):

                    ner_dict = format_label_fields(ner)
                    sentence, spans, ner_labels = self.text_to_instance(sentence, ner_dict)
                    processed.append([sentence, spans, ner_labels])

        else:  #serving
            for sentence in data:
                sentence, spans = self.text_to_instance(sentence, None, mode=mode)

                processed.append([sentence, spans])
                
        return processed

    def text_to_instance(self,
                        sentence,
                        ner_dict,
                        mode='training'
                        ):
        spans = []
        ner_labels = []
        if mode == 'training':
            for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
                span_ix = (start, end)
                spans.append((start, end))
                ner_label = ner_dict[span_ix]
                ner_labels.append(self.ner_label.get_id(ner_label))
            
            return sentence, spans, ner_labels

        else:
            for start, end in self.enumerate_spans(sentence, max_span_width=self.max_span_width):
                span_ix = (start, end)
                spans.append((start, end))

            return sentence, spans
            
    def enumerate_spans(self, sentence, max_span_width, min_span_width=1):

        max_span_width = max_span_width or len(sentence)
        spans = []

        for start_index in range(len(sentence)):
            last_end_index = min(start_index + max_span_width, len(sentence))
            first_end_index = min(start_index + min_span_width - 1, len(sentence))
            for end_index in range(first_end_index, last_end_index):
                start = start_index
                end = end_index
                spans.append((start, end))
        return spans


    def __len__(self):
        return 100000000

    def __getitem__(self, index):

        index = random.randint(0, len(self.training_data)-1)

        raw_data = self.training_data[index]
        data = raw_data.deepcopy_all_data()
        
        tokens, idx_dict = self.encoder.tokenize(data['sentence'])
        
        converted_spans = []
        for span in data['spans']:
            converted_spans.append(self.convert_span(span, idx_dict))

        data['sentence'] = tokens
        data['spans'] = converted_spans
             
        return [raw_data, data]


    def get_unlabel_item(self, index):

        raw_data = self.unlabel_data[index]
        data = raw_data.deepcopy_all_data()

        tokens, idx_dict = self.encoder.tokenize(data['sentence'])
        
        converted_spans = []
        for span in data['spans']:
            converted_spans.append(self.convert_span(span, idx_dict))

        data['sentence'] = tokens
        data['spans'] = converted_spans
             
        return [raw_data, data]

    
    def convert_span(self, span, idx_dict):

        start_idx = span[0]
        end_idx = span[1]
        
        span_idx = idx_dict[start_idx] + idx_dict[end_idx]

        if len(span_idx)==0:  ## some special character in Bert tokenizer will become None, like white space in unicode
            return (0, 0)

        return (min(span_idx), max(span_idx))

    def collate_fn(self, data):
        
        raw_data_b, data_b = zip(*data)
        data_b = list_to_dict(data_b)
        
        max_length = max([len(tokens) for tokens in data_b['sentence']])
        ##padding
        for tokens in data_b['sentence']:
            while len(tokens)<max_length:
                tokens.append(0)
        
        data_b['sentence'] = torch.LongTensor(data_b['sentence'])
        ##mask
        data_b['mask'] = data_b['sentence'].eq(0).eq(0).float()


        span_max_length = max([len(converted_spans) for converted_spans in data_b['spans']]) or 1 ## at least length is 1
        ##span padding
        for converted_spans in data_b['spans']:
            while len(converted_spans)<span_max_length:
                converted_spans.append((0, 0))
        
        data_b['spans'] = torch.LongTensor(data_b['spans'])
        ## span label padding
        for ner_labels in data_b['ner_labels']:
            while len(ner_labels)<span_max_length:
                ner_labels.append(0)
                
        data_b['ner_labels'] = torch.LongTensor(data_b['ner_labels'])

        for soft_labels in data_b['soft_labels']:
            while len(soft_labels)<span_max_length:
                soft_labels.append([0]*self.ner_label.get_num())
        
        data_b['soft_labels'] = torch.FloatTensor(data_b['soft_labels'])
        
        ##span mask
        
        for span_mask in data_b['span_mask']:
            while len(span_mask)<span_max_length:
                span_mask.append(0)

        data_b['span_mask'] = torch.FloatTensor(data_b['span_mask'])

        for span_mask_for_loss in data_b['span_mask_for_loss']:
            while len(span_mask_for_loss)<span_max_length:
                span_mask_for_loss.append(0)
        
        data_b['span_mask_for_loss'] = torch.FloatTensor(data_b['span_mask_for_loss'])

        return raw_data_b, data_b

    def update_dataset(self, new_data):

        self.training_data = deepcopy(self.label_data) + new_data

class MyDataLoader: ## For unlabeled data
    def __init__(self, dataset, batch_size):

        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.max_batch_i = len(dataset.unlabel_data)//batch_size
        self.max_one_i = len(dataset.unlabel_data)
    
    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.max_one_i
    
    def __next__(self):
        
        if len(self.dataset.unlabel_data)==0:  ## empty dataset check
            raise StopIteration

        batch_data = []
        if self.index < self.max_batch_i:
            for i in range(self.index * self.batch_size, (self.index+1) * self.batch_size):
                batch_data.append(self.dataset.get_unlabel_item(i))
            
        elif self.index == self.max_batch_i:
            for i in range(self.index * self.batch_size, self.max_one_i):
                batch_data.append(self.dataset.get_unlabel_item(i))
        else:
            raise StopIteration
        self.index+=1
        return self.dataset.collate_fn(batch_data)
    
    def reset(self):
        self.index = 0

    def has_next(self):
        
        return self.index * self.batch_size < self.max_one_i


def get_loader(root, filename, encoder, batch_size, ner_label, is_train, opt, mode='training'):

    dataset = DataSet(root, filename, encoder, batch_size, ner_label, is_train=is_train, opt=opt, mode=mode)

    label_data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=dataset.collate_fn)

    unlabel_data_loader = MyDataLoader(dataset=dataset, batch_size=batch_size)

    if is_train:
        return iter(label_data_loader), iter(unlabel_data_loader)
    else:
        return iter(label_data_loader)

def update_train_loader(dataset, batch_size):

    label_data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=dataset.collate_fn)

    unlabel_data_loader = MyDataLoader(dataset=dataset, batch_size=batch_size)

    return iter(label_data_loader), iter(unlabel_data_loader)
