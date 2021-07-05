import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, lexical_dropout, lower): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.lower = lower

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        
    def forward(self, tokens, mask):
        
        outputs = self.bert(tokens, attention_mask=mask)
        outputs = self._lexical_dropout(outputs.last_hidden_state)

        return outputs
    
    def tokenize(self, raw_tokens):
        # token -> index
           
        tokens = ['[CLS]']
        idx_dict = dict()
        cur_pos = 1

        for i, token in enumerate(raw_tokens):
            
            if self.lower:
                input_token = token.lower()
            else:
                input_token = token

            sub_words = self.tokenizer.tokenize(input_token)
            tokens += sub_words
            idx_dict[i] = list(range(cur_pos, cur_pos+len(sub_words)))
            cur_pos += len(sub_words)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens, idx_dict

    def tokenize_to_string(self, raw_tokens):  ## for serving

        # token -> index
           
        tokens = ['[CLS]']

        for i, token in enumerate(raw_tokens):
            
            if self.lower:
                input_token = token.lower()
            else:
                input_token = token

            sub_words = self.tokenizer.tokenize(input_token)
            tokens += sub_words

        return tokens

    def freeze(self):

        for param in self.bert.base_model.parameters():
            param.requires_grad = False

    def encode_instance(self, instance, iner=False):

        sentence = instance.sentence
        span = instance.span
        

        tokens, idx_dict = self.tokenize(sentence)
        sentence_length = len(tokens)
        span_idx1 = idx_dict[span[0]][0]
        span_idx2 = idx_dict[span[1]][-1]
        tokens = torch.LongTensor(tokens).view(1, sentence_length)

        if torch.cuda.is_available():
            tokens = tokens.cuda()

        outputs = self.bert(tokens)
        hidden = outputs.hidden_states
        outputs = torch.stack(hidden[:3]).mean(dim=0)
        span_embedding = torch.mean(outputs.squeeze()[span_idx1:span_idx2+1,:], dim=0)

        if not iner:

            span_embedding = span_embedding.cpu().detach().numpy()

        return span_embedding

    def encode_instances(self, instances):

        instance_embeddings = []
        for instance in instances:

            instance_embeddings.append(self.encode_instance(instance, True))

        rule_embedding = torch.mean(torch.stack(instance_embeddings, dim=0), dim=0).cpu().detach().numpy()

        return rule_embedding