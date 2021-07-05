import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import EndpointSpanExtractor, SelfAttentiveSpanExtractor, FeedForward
from tallor.categorical_accuracy import CategoricalAccuracy
from tallor.precision_recall_f1 import PrecisionRecallF1
from tallor.utils import sequence_cross_entropy_with_logits, sequence_cross_entropy_with_soft_label, get_num_spans_to_keep, prob_mask

class JointIE(nn.Module):
    def __init__(self, 
                sentence_encoder, 
                hidden_size, 
                embedding_size, 
                ner_label,
                context_layer, 
                context_dropout=0.3,
                dropout = 0.3,
                span_repr_combination = 'x,y',
                max_span_width = 5,
                span_width_embedding_dim = 64,
                spans_per_word = 0.6,
                use_soft_label = True
                ):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.contextual_layer = nn.LSTM(embedding_size, embedding_size//2, num_layers=context_layer, bidirectional=True, dropout=context_dropout)

        self.endpoint_span_extractor = EndpointSpanExtractor(embedding_size,
                                                                combination=span_repr_combination,
                                                                num_width_embeddings=max_span_width * 100,
                                                                span_width_embedding_dim=span_width_embedding_dim,
                                                                bucket_widths=False)

        self.attentive_span_extractor = SelfAttentiveSpanExtractor(embedding_size)
        self.ner_label = ner_label
        ## span predictioin layer
        span_emb_dim = self.attentive_span_extractor.get_output_dim() + self.endpoint_span_extractor.get_output_dim()
        ner_label_num = ner_label.get_num()
        self.span_layer = FeedForward(input_dim = span_emb_dim, num_layers=2, hidden_dim=hidden_size, dropout=dropout)
        self.span_proj_label = nn.Linear(hidden_size, ner_label_num)

        self.spans_per_word = spans_per_word
        self.ner_neg_id = ner_label.get_neg_id()
        self.use_soft_label = use_soft_label

        ## metrics
        # ner
        self.ner_acc = CategoricalAccuracy(top_k=1, tie_break=False)
        self.ner_prf = PrecisionRecallF1(neg_label=self.ner_neg_id)
        self.ner_prf_b = PrecisionRecallF1(neg_label=self.ner_neg_id, binary_match=True)

                                                                    
    def forward(self, 
                tokens, # (batch_size, length)
                mask,   # (batch_size, length)
                converted_spans,  # (batch_size, span_num, 2)
                span_mask,        # (batch_size, span_num)
                span_mask_for_loss,
                ner_labels,       # (batch_size, span_num)
                soft_labels,       #(batch_size, span_num, class_num)
                ):
        
        seq_len = tokens.size(1)

        embedding = self.sentence_encoder(tokens, mask) # (batch_size, length, bert_dim)
        contextual_embedding, _ = self.contextual_layer(embedding) #(batch_size, length, hidden_size)

        # extract span representation
        ep_span_emb = self.endpoint_span_extractor(contextual_embedding, converted_spans, span_mask)  #(batch_size , span_num, hidden_size)
        att_span_emb = self.attentive_span_extractor(embedding, converted_spans, span_mask)   #(batch_size, span_num, bert_dim)
        #span_emb = att_span_emb
        span_emb = torch.cat((ep_span_emb, att_span_emb), dim = -1)  #(batch_size, span_num, hidden_size+bert_dim)

        span_logits = self.span_proj_label(self.span_layer(span_emb))  #(batch_size, span_num, span_label_num)
        span_prob = F.softmax(span_logits, dim=-1)  #(batch_size, span_num, span_label_num)
        _, span_pred = span_prob.max(2)

        span_prob_masked = prob_mask(span_prob, span_mask)  #(batch_size, span_num, span_label_num)

        if self.training:
            num_spans_to_keep = get_num_spans_to_keep(self.spans_per_word, seq_len, span_prob.size(1))
            top_v = (-span_prob_masked[:, :, self.ner_neg_id]).topk(num_spans_to_keep, -1)[0][:, -1:]
            top_mask = span_prob[:, :, self.ner_neg_id] <= -top_v  #(batch_size, span_num)
            span_mask_subset = span_mask * (top_mask | ner_labels.ne(self.ner_neg_id)).float()
            span_mask_loss = span_mask_subset * span_mask_for_loss
        else:
                
            span_mask_subset = span_mask
            span_mask_loss = span_mask

        if self.use_soft_label:
            span_loss = sequence_cross_entropy_with_soft_label(
                        span_logits, soft_labels, span_mask_loss,
                        average='sum')
        else:
            span_loss = sequence_cross_entropy_with_logits(
                        span_logits, ner_labels, span_mask_loss,
                        average='sum')

        span_len = converted_spans[:, :, 1] - converted_spans[:, :, 0] + 1

        ## span metrics
        self.ner_acc(span_logits, ner_labels, span_mask_subset)
        self.ner_prf(span_logits.max(-1)[1], ner_labels, span_mask_subset.long(), bucket_value=span_len)
        self.ner_prf_b(span_logits.max(-1)[1], ner_labels, span_mask_subset.long())

        ## loss
        loss = span_loss

        ## output dict

        output_dict = {
            'loss': loss,
            'span_loss': span_loss,
            'span_pred': span_pred,
            'span_metrics': [self.ner_acc, self.ner_prf, self.ner_prf_b]
        }

        return output_dict

    def predict(self, 
                tokens, # (batch_size, length)
                mask,   # (batch_size, length)
                converted_spans,  # (batch_size, span_num, 2)
                span_mask,        # (batch_size, span_num)
                ):

        embedding = self.sentence_encoder(tokens, mask) # (batch_size, length, bert_dim)
        contextual_embedding, _ = self.contextual_layer(embedding) #(batch_size, length, hidden_size)

        # extract span representation
        ep_span_emb = self.endpoint_span_extractor(contextual_embedding, converted_spans, span_mask)  #(batch_size , span_num, hidden_size)
        att_span_emb = self.attentive_span_extractor(embedding, converted_spans, span_mask)   #(batch_size, span_num, bert_dim)
        #span_emb = att_span_emb
        span_emb = torch.cat((ep_span_emb, att_span_emb), dim = -1)  #(batch_size, span_num, hidden_size+bert_dim)

        span_logits = self.span_proj_label(self.span_layer(span_emb))  #(batch_size, span_num, span_label_num)
        span_prob = F.softmax(span_logits, dim=-1)  #(batch_size, span_num, span_label_num)
        _, span_pred = span_prob.max(2)

        
        output_dict = {
            'tokens': tokens,
            'spans': converted_spans,
            'span_mask': span_mask,
            'span_prob': span_prob,
            'span_pred': span_pred
        }

        return output_dict

    def decode(self, output_dict):
        '''
        Decode predictions to training labels and compute cofidence score for each instance.
        '''
        ## for ner 
        spans_batch = output_dict['spans'].detach().cpu()  ## (batch, span_num, 2)
        predicted_ner_batch = output_dict['span_pred'].detach().cpu()  ## (batch, span_num)
        span_mask_batch = output_dict['span_mask'].detach().cpu().bool() ## (batch_size, span_num)
        span_prob_batch = output_dict['span_prob'].detach().cpu()  ## (batch_size, span_num, span_label_num)

        ner_res_list = self.ner_decode(spans_batch, predicted_ner_batch, span_mask_batch, span_prob_batch)

        return ner_res_list

        

    def ner_decode(self, spans_batch, predicted_ner_batch, span_mask_batch, span_prob_batch):

        res_list = []
        
        for spans, span_mask, predicted_NERs, span_prob in zip(spans_batch, span_mask_batch, predicted_ner_batch, span_prob_batch):
            entry_list = []
    
            for i, (span, mask, ner, prob) in enumerate(zip(spans, span_mask, predicted_NERs, span_prob)):
                ner = ner.item()
                mask = mask.item()
                if mask!=0:
                    the_span = (span[0].item(), span[1].item())
                    entry_list.append({'span_idx': i, 
                                        'span': the_span, 
                                        'prob': prob[ner].item(), 
                                        'class': ner})

            res_list.append(entry_list)

        return res_list

    def metric_reset(self):

        self.ner_acc.reset()
        self.ner_prf.reset()
        self.ner_prf_b.reset()

