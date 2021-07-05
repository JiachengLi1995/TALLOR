import os
import sys
from .data_loader import update_train_loader
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from copy import deepcopy
from collections import defaultdict
from tallor.rule_kits.rule_labeler import RuleLabeler
from tqdm import tqdm

class IEFramework:

    def __init__(self, ner_label, dataset_name, opt, logger, batch_size, train_data_loader, val_data_loader, test_data_loader, unlabel_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.unlabel_data_loader = unlabel_data_loader
        self.training_set = self.unlabel_data_loader.dataset
        self.batch_size = batch_size
        self.Labeler = RuleLabeler(ner_label, self.training_set.unlabel_data, dataset_name, opt)
        self.logger = logger
        
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def train(self,
              model,
              model_name,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=100,
              update_threshold=0.7
              ):
            
        # Init
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)

        self.save_initial_model(model, save_ckpt)
     
        # Training
        best_ner_f1 = 0        
        self.logger.info('Start training!')

        for i in range(train_iter//val_step):

            rule_labeled_data, all_data = self.Labeler.pipeline(i)  # label by dictionary
            self.Labeler.eval(i, all_data, rule_labeled_data)  ## evaluate the rule performance for experiments, comment it for unlabeled data.

            self.update_dataset_and_loader(self.training_set, rule_labeled_data)
            self.load_initial_model(model, save_ckpt)
            train_step = val_step + 50*i
            train_f1, train_p, train_r = self.train_ner_model(model, train_step, warmup_step, best_ner_f1)
            ner_f1, ner_p, ner_r = self.eval(model, val_iter, i)

            if ner_f1 > best_ner_f1:
                print('Best checkpoint')
                torch.save({'state_dict': model.state_dict()}, save_ckpt)
                best_ner_f1 = ner_f1
                
            self.select_and_update_training(model, update_threshold)
            model.metric_reset()

        model.metric_reset()
        print("\n####################\n")
        self.logger.info("Finish training " + model_name)
    
    def train_ner_model(self, model, train_iter, warmup_step, best_ner_f1):

        model.train()

        parameters_to_optimize = []
        for n, p in list(model.named_parameters()):
            if p.requires_grad:
                parameters_to_optimize.append((n, p))
        

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        for it in tqdm(range(train_iter), ncols=100, total=train_iter, desc='Train NER model.'):
            
            raw_data_b, data_b = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    data_b[k] = v.cuda()
            
            output_dict  = model(data_b['sentence'], data_b['mask'], data_b['spans'], data_b['span_mask'], 
                                data_b['span_mask_for_loss'], data_b['ner_labels'], data_b['soft_labels'])
            
            loss = output_dict['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


            ner_results = output_dict['span_metrics']
           
            ner_acc = ner_results[0].get_metric()
            ner_prf = ner_results[1].get_metric()
            ner_prf_b = ner_results[2].get_metric()
            
            # sys.stdout.write('step: {0:4} | loss: {1:2.6f},  NER_acc: {2:3.2f}%'.format(it + 1, loss, 100 * ner_acc) +'\n')
            # sys.stdout.write('NER \t F1 \t Precision \t Recall \n')
            # sys.stdout.write('prf \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} '.format(ner_prf['f'], ner_prf['p'], ner_prf['r']) +'\n')
            # sys.stdout.write('prf_b \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(ner_prf_b['f'], ner_prf_b['p'], ner_prf_b['r']) +'\n')
            # sys.stdout.write('Current Best on valid. NER F1: {0:2.4f} \t'.format(best_ner_f1) +'\n')
            # sys.stdout.flush()
        
        return ner_prf['f'], ner_prf['p'], ner_prf['r']

    def eval(self,
            model,
            eval_iter,
            step,
            ckpt=None): 

        model.metric_reset()
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        with torch.no_grad():
            for it in tqdm(range(eval_iter), ncols=100, total=eval_iter, desc='Evaluation.'):
                
                raw_data_b, data_b = next(eval_dataset)
        
                if torch.cuda.is_available():
                    for k, v in data_b.items():
                        data_b[k] = v.cuda()

                output_dict  = model(data_b['sentence'], data_b['mask'], data_b['spans'], data_b['span_mask'], 
                                data_b['span_mask_for_loss'], data_b['ner_labels'], data_b['soft_labels'])

                loss = output_dict['loss']

                ner_results = output_dict['span_metrics']
                
                ner_acc = ner_results[0].get_metric()
                ner_prf = ner_results[1].get_metric()
                ner_prf_b = ner_results[2].get_metric()

                # sys.stdout.write('step: {0:4} | loss: {1:2.6f},  NER_acc: {2:3.2f}%'.format(it + 1, loss, 100 * ner_acc) +'\n')
                # sys.stdout.write('NER \t F1 \t Precision \t Recall \n')
                # sys.stdout.write('prf \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} '.format(ner_prf['f'], ner_prf['p'], ner_prf['r']) +'\n')
                # sys.stdout.write('prf_b \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(ner_prf_b['f'], ner_prf_b['p'], ner_prf_b['r']) +'\n')
                # sys.stdout.flush()
                
        self.logger.info('STEP:{0}. Current performance on NER: F1: {1:2.4f}, P: {2:2.4f}, R: {3:2.4f}.'.format(step, ner_prf['f'], ner_prf['p'], ner_prf['r']))
        self.logger.info('Binary result: F1: {0:2.4f}, P: {1:2.4f}, R: {2:2.4f}.'.format(ner_prf_b['f'], ner_prf_b['p'], ner_prf_b['r']))
        return ner_prf['f'], ner_prf['p'], ner_prf['r']


    def select_and_update_training(self, model, update_threshold):

        model.eval()         
        new_data_self_training, raw_data = self.self_training(model, update_threshold)
        
        self.Labeler.update_rule_pipeline(new_data_self_training)

    def update_dataset_and_loader(self, dataset, new_data):
    
        dataset.update_dataset(new_data)

        del self.train_data_loader

        self.train_data_loader, self.unlabel_data_loader = update_train_loader(dataset, self.batch_size)

        self.logger.info(f'Update successfully! {len(dataset.training_data)} instances for training, {len(new_data)} instances are new labeled.')


    def self_training(self, model, update_threshold):

        raw_data = []
        ner_res = []
        print('Begin predict all data.')
        while self.unlabel_data_loader.has_next():
            
            raw_data_b, data_b = next(self.unlabel_data_loader)
            
            if torch.cuda.is_available():
                for k, v in data_b.items():
                    data_b[k] = v.cuda()
            
            output_dict  = model.predict(data_b['sentence'], data_b['mask'], data_b['spans'], data_b['span_mask'])
            
            ner_res_list = model.decode(output_dict)
            
            raw_data += raw_data_b
            ner_res += ner_res_list
        print('Done.')
        new_data = self.select_and_update_data(raw_data, ner_res, update_threshold)

        return new_data, raw_data

    def select_and_update_data(self, raw_data, ner_res, update_threshold):

        ner_res_dict = defaultdict(list) # key is the class, value is the res instances

        data_id_ner_dict = defaultdict(list)  # key is the data_id, value is the res intances (for update relation)
    
        for i, ner_res_entry in enumerate(ner_res):

            for ner in ner_res_entry:
                ner['data_id'] = i
                ner_res_dict[ner['class']].append(ner)
        
        for value in ner_res_dict.values():
            value = sorted(value, key=lambda x: x['prob'], reverse=True)
            selected_ner_res = value[:int(len(value)*update_threshold)]
            for instance in selected_ner_res:
                data_id_ner_dict[instance['data_id']].append(instance)

        new_data = []
        new_raw_data = deepcopy(raw_data)
        for i, data_entry in enumerate(new_raw_data):

            if i in data_id_ner_dict:
                new_data_entry = self.update_data_entry(data_entry, data_id_ner_dict[i])
            else:
                new_data_entry = data_entry

            new_data.append(new_data_entry)
        return new_data

    
    def update_data_entry(self, data_entry, ner_res_list):
    
        if len(ner_res_list)>0:

            for ner_res in ner_res_list:
                span_idx = ner_res['span_idx']
                label = ner_res['class']
                data_entry.ner_labels[span_idx] = label
                data_entry.span_mask_for_loss[span_idx] = 1

        return data_entry

    def save_initial_model(self, model, path):

        torch.save({'state_dict': model.state_dict()}, path+'_initial')

    def load_initial_model(self, model, path):

        state_dict = self.__load_model__(path+'_initial')['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)




            



        
        








