import torch
import math
from copy import deepcopy
from collections import defaultdict

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.id2label = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.id2label[self.label_num] = label
        self.label_num += 1

        return self.label2id[label]

    def get_label(self, id):

        if id not in self.id2label:
            print(f'Cannot find label that id is {id}!!!')
            assert 0
        return self.id2label[id]
    
    def get_num(self):
        return self.label_num

    def is_exist(self, label):

        return label in self.label2id

    def get_neg_id(self):

        return self.get_id('')
    
    def all_labels(self):
        return list(self.label2id.keys())

    def get_soft_label(self, label):

        if isinstance(label, int):
            soft_label = [0]*self.label_num
            soft_label[label] = 1
            return soft_label
        elif isinstance(label, str):
            label_id = self.get_id(label)
            soft_label = [0]*self.label_num
            soft_label[label_id] = 1
            return soft_label
        else:
            NotImplementedError

class DataPoint:
    def __init__(self, 
                sentence,
                spans, 
                ner_labels,
                parsed_tokens,
                label_num
                ):
        
        self.sentence = sentence
        self.spans = spans 
        self.ner_labels = ner_labels
        self.ground_truth = deepcopy(ner_labels)
        self.span_mask = [1]*len(spans)
        self.span_mask_for_loss = [1]*len(spans)

        self.soft_labels = [[0]*label_num for label in self.ner_labels]

        self.parsed_tokens = parsed_tokens

        assert len(parsed_tokens) == len(sentence)

    def deepcopy_all_data(self):
        
        return deepcopy({'sentence': self.sentence,
                'spans': self.spans,
                'ner_labels': self.ner_labels,
                'soft_labels': self.soft_labels,
                'span_mask':self.span_mask,
                'span_mask_for_loss': self.span_mask_for_loss
                })

    def unlabel_reset(self):

        self.span_mask_for_loss = [0] * len(self.spans)

def list_to_dict(L):
    '''
    Convert a list of dict to a dict
    Example:
    [dict, dict, dict] -> dict{
        k1: list,
        k2: list,
        k3: list
    }
    '''
    new_dict = defaultdict(list)
    for d in L:
        for k, v in d.items():
            new_dict[k].append(v)
    
    return new_dict

def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.FloatTensor:
    if average not in {None, "token", "batch", "sum", "batch_sum"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', 'batch', 'sum', or 'batch_sum'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    elif average == "sum":
        return negative_log_likelihood.sum()
    elif average == "batch_sum":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss.sum()
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss

def sequence_cross_entropy_with_soft_label(logits: torch.FloatTensor,
                                            targets: torch.FloatTensor, #(batch, sequence_length, class_num)
                                            weights: torch.FloatTensor,
                                            average: str = "batch"
                                            ) -> torch.FloatTensor:
    if average not in {None, "token", "batch", "sum", "batch_sum"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', 'batch', 'sum', or 'batch_sum'")

    # num_classes = logits.size(-1)
    # # shape : (batch * sequence_length, num_classes)
    # logits_flat = logits.view(-1, num_classes)
    # # shape : (batch * sequence_length, num_classes)
    # log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # # shape : (batch * sequence_length, num_classes)
    # targets_flat = targets.view(-1, num_classes)

    # negative_log_likelihood_flat = - log_probs_flat * targets_flat
    # negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    
    # # shape : (batch, sequence_length)
    # negative_log_likelihood = negative_log_likelihood_flat.view(*weights.size())
    # # shape : (batch, sequence_length)
    # negative_log_likelihood = negative_log_likelihood * weights.float()

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1, logits.size(-1))
    num_points, num_classes = logits_flat.shape
    cum_losses = logits_flat.new_zeros(num_points)

    for y in range(num_classes):
        target_temp = logits_flat.new_full((num_points,), y, dtype=torch.long)
        y_loss = torch.nn.functional.cross_entropy(logits_flat, target_temp, reduction="none")
        cum_losses += targets_flat[:, y].float() * y_loss

    negative_log_likelihood = cum_losses.view(*weights.size())
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    elif average == "sum":
        return negative_log_likelihood.sum()
    elif average == "batch_sum":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss.sum()
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


def prob_mask(prob: torch.FloatTensor,
            mask: torch.FloatTensor,
            value: float = 1.0):
    ''' Add value to the positions masked out. prob is larger than mask by one dim. '''
    return prob + ((1.0 - mask) * value).unsqueeze(-1)

def get_num_spans_to_keep(spans_per_word,
                        seq_len,
                        max_value):

    if type(spans_per_word) is float:
        num_spans_to_keep = max(min(int(math.floor(spans_per_word * seq_len)), max_value), 1)
    elif type(spans_per_word) is int:
        num_spans_to_keep = max(min(spans_per_word, max_value), 1)
    else:
        raise ValueError
    return num_spans_to_keep

