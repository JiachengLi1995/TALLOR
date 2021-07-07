# TALLOR

## Publication
Please cite the following paper if you are using TALLOR. Thanks!

* Jiacheng Li, Haibo Ding, Jingbo Shang, Julian McAuley, Zhe Feng. [Weakly Supervised Named Entity Tagging with Learnable Logical Rules](https://arxiv.org/abs/2107.02282). (ACL 2021)

## Recommended Environment

```bash
pip install -r requirements.txt
```

## Experiments in Paper
In this section, we introduce how to reimplement the experiments in our paper. We already include all needed datasets and rule files in this repo.
### Dataset
3 datasets are preprocessed and included in this repository.

| Dataset             | Task code     | Dir                      | Source   |
|---------------------|---------------|--------------------------|----------|
| BC5CDR   | bc5cdr           | data/bc5cdr                 | [link](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/)|
| CHEMDNER              | chemdner        | data/chemdner             | [link](https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/)|
| CoNLL 2003              | conll2003         | data/conll2003               | [link](https://arxiv.org/pdf/cs/0306050v1.pdf)|


### Train and Evaluate Models
```
python train_demo.py --dataset bc5cdr --encoder scibert
```

### Outputs
We have examples of output files of experiments on BC5CDR dataset and we also describe these output files here.
| Filename or Path | Descriptions |
|----------|--------------|
|checkpoint/| Best checkpoint of neural model.|
|logging/JointIE.log| Evaluation on dev and test dataset via iterations.|
|logging/RuleSelector.log| Selected rules of each iteration.|
|logging/InstanceSelector.log| Thresholds and scores of dynamic instance selection.|

## Serve on Customer's Text
In this Section, we introduce how to run Tallor on your own plain text dataset to recognize the entities starting only from a few rules. We include a dataset ```bc5cdr_serving``` as an example here.

### Step 1: Prepare your dataset
* Please prepare your text data into a ```.json``` file as following and put it in ```data/[your dataset name]/[your file name].json```: (example file is ```data/bc5cdr_serving/serving.json```)
```
{"sentence": ["A", "lesser", "degree", "of", "orthostatic", "hypotension", "occurred", "with", "standing", "."]}
```
* Put the multi-gram phrases file generated from **[AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase)** into ```data/[your dataset name]/AutoPhrase_multi-words.txt```.
(example file is ```data/bc5cdr_serving/AutoPhrase_multi-words.txt```)

### Step 2: Write your own rules
* You can write your own rules for your dataset in file ```tallor/label_functions/serving_template.py``` as a Python ```dict``` as following (we already have rules for BC5CDR dataset in this file, please comment it and write your own rules):
```
dictionary = {'proteinuria': 'Disease', 'esrd': 'Disease', 'thrombosis': 'Disease', 'tremor': 'Disease', 'hepatotoxicity': 'Disease','nicotine': 'Chemical', 'morphine': 'Chemical', 'haloperidol': 'Chemical', 'warfarin': 'Chemical', 'clonidine': 'Chemical'}
```
### Step 3: Run TALLOR
* Run our model TALLOR and please check the hyperparameters in the next section:
```
python serving.py --dataset [your dataset name]

Example: python serving.py --dataset bc5cdr_serving --filename serving.json --encoder scibert
```

### Step 4: Get your results
* Extracted rules and recognized entities are saved into path ```serving/[your dataset name]``` and include two files ```extracted_rules.json``` and ```ner_results.json```.

## Key Hyperparameter Table in Serving
We believe that our default parameters can help you get a good start to tune the hyperparameters. Please refer to the table to select your parameters.
| Parameters | Description|
|------------|------------|
|--filename| Your dataset file.|
|--dataset| Your dataset directory.|
|--encoder| Pre-trained langauge used in our model ```bert``` or ```scibert```.|
|--epoch|Number of iterations. Empirically, model with **Larger** epoch has results with better **Recall**, **Smaller** means better **Precision**.|
|--train_step| Number of training steps of neural model in each iteration. This number will increase 50 after one epoch.|
|--update_threshold|The ratio of the most confident data used for evaluateing and updating new rules.|
|--rule_threshold|The minimum frequency of rules.|
|--rule_topk|Number of rules in each entity category are selected as new rules in each epoch.|
|--global_sample_times|Sample times for global scores.|
|--threshold_sample_times|Sample times for computing dynamic threshold.|
|--temperature| Temperature to control threshold. Larger will have more strict instance selection strategy.|

## Contact
Jiacheng Li

E-mail: j9li@eng.ucsd.edu
