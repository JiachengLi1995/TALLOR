from tallor.data_loader import get_loader
from tallor.framework_serving import IEFramework
from tallor.sentence_encoder import BERTSentenceEncoder
from tallor.utils import LabelField
from tallor.label_functions.LF_manager import LFManager
from models.JointIE import JointIE
import torch
import sys
import argparse
import os
import logging
import time

def main():
    parser = argparse.ArgumentParser()
    ## File parameters
    parser.add_argument('--filename', default='serving.json',
            help='train file')
    parser.add_argument('--dataset', default='bc5cdr_serving',
            help='dataset')
    parser.add_argument('--root', default='./data',
            help='dataset root')

    ## span
    parser.add_argument('--max_span_width', default=5, type=int,
            help='max number of word in a span')
    
    ## encoder
    parser.add_argument('--lexical_dropout', default=0.5, type=float,
            help='Embedding dropout')
    parser.add_argument('--embedding_size', default=768, type=float,
            help='Embedding dimension')
    parser.add_argument('--lower', default=1, type=int,
            help='1: lower case  0: upper case')
    parser.add_argument('--freeze', action='store_true',
            help='freeze bert model')
    
    ## model
    parser.add_argument('--model', default='JointIE',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: bert or scibert')
    parser.add_argument('--hidden_size', default=512, type=int,
           help='hidden size')
    parser.add_argument('--context_layer', default=1, type=int,
           help='number of contextual layers')
    parser.add_argument('--context_dropout', default=0, type=int,
           help='dropout rate in the contextual layer')
    parser.add_argument('--dropout', default=0.3, type=float,
           help='dropout rate')
    parser.add_argument('--span_width_dim', default=64, type=int,
           help='dimension of embedding for span width')
    parser.add_argument('--spans_per_word', default=0.6, type=float,
           help='thershold number of spans in each sentence')

    ## Train
    parser.add_argument('--batch_size', default=32, type=int,
            help='batch size')
    parser.add_argument('--epoch', default=20, type=int,
            help='num of epoch in training')
    parser.add_argument('--train_step', default=1500, type=int,
            help='num of training step for neural model.')
    parser.add_argument('--warmup_step', default=100, type=int,
           help='warmup steps for neural tagger')
    parser.add_argument('--update_threshold', default=0.7, type=float,
           help='the ratio of the most confident data used for evaluateing and updating new rules')
    parser.add_argument('--labeled_ratio', default=0, type=float,
           help='The ratio of labeled data used for training.')
    parser.add_argument('--not_use_soft_label', action='store_false',
           help='Do not use soft label for training.')

    ## Rules
    parser.add_argument('--rule_threshold', default=2, type=int,
            help='Rule frequency threshold.')
    parser.add_argument('--rule_topk', default=20, type=int,
            help='Select topk rules added to rule set.')

    ## Instance Selector
    parser.add_argument('--global_sample_times', default=50, type=int,
            help='Sample times for global scores.')
    parser.add_argument('--threshold_sample_times', default=100, type=int,
            help='Sample times for computing dynamic threshold.')
    parser.add_argument('--temperature', default=0.8, type=float,
            help='Temperature to control threshold.')       
    ## Save
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    parser.add_argument('--serving_output', default='./serving')
 
    opt = parser.parse_args()
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    opt.lower = bool(opt.lower)
    root = os.path.join(opt.root, opt.dataset)
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))

    ## set sentence encoder for neural tagger       
    if encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        opt.embedding_size = 768
        opt.lower = 'uncased' in pretrain_ckpt
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, opt.lower)
    elif encoder_name == 'scibert':
        pretrain_ckpt = opt.pretrain_ckpt or 'allenai/scibert_scivocab_uncased'
        opt.embedding_size = 768
        opt.lower = 'uncased' in pretrain_ckpt
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, opt.lexical_dropout, opt.lower)
    else:
        raise NotImplementedError

    if opt.freeze:
        sentence_encoder.freeze()
    
    ner_label = LabelField()
    LFManager.set_ner_label(ner_label)
    
    ## read dataset
    dataset_name = os.path.basename(root)
    train_data_loader, unlabel_data_loader = get_loader(root, opt.filename, sentence_encoder, batch_size, ner_label, is_train=True, opt=opt, mode='serving')

    ## set logger
    prefix = '-'.join([model_name, encoder_name, opt.dataset, opt.filename, str(opt.labeled_ratio), str(time.time())])
    logger = set_logger(prefix)
    logger.info(opt)
    framework = IEFramework(ner_label, dataset_name, opt, logger, batch_size, train_data_loader, unlabel_data_loader)
        
    if model_name == 'JointIE':
        model = JointIE(sentence_encoder, opt.hidden_size, opt.embedding_size, ner_label, 
                        opt.context_layer, opt.context_dropout, opt.dropout,
                        max_span_width=opt.max_span_width, span_width_embedding_dim=opt.span_width_dim,
                        spans_per_word=opt.spans_per_word, use_soft_label=opt.not_use_soft_label)
        if torch.cuda.is_available():
            model.cuda()

    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not os.path.exists(opt.serving_output):
        os.mkdir(opt.serving_output)
    opt.serving_output = os.path.join(opt.serving_output, opt.dataset)
    if not os.path.exists(opt.serving_output):
        os.mkdir(opt.serving_output)

    framework.train(model, prefix, epoch=opt.epoch, train_step=opt.train_step, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                    warmup_step=opt.warmup_step, update_threshold=opt.update_threshold, result_dir=opt.serving_output)

def set_logger(prefix):
    ## set logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('WeakNER')

    if not os.path.exists('./logging'):
        os.mkdir('./logging')

    file_handler = logging.FileHandler('./logging/'+prefix+'.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger

if __name__ == "__main__":
    main()
