# ICSME2022CrossCS
Code and data for ICSME 2022 research track paper "Cross-Modal Contrastive Learning for Code Search"

### Dataset
You can obtain the dataset directly from this [link](https://drive.google.com/drive/folders/18uJ7bkDrVo86HlOqegI01S4LY3FoO8q1?usp=sharing).

### Dependency
- python>=3.6
- torch==1.4.0
- transformers>=2.5.0

### Fine-tune
```shell
sh run.sh
```

### Inference
```shell
sh test.sh
```

### Evaluator
```shell
python evaluator/evaluator.py -a dataset/test.jsonl  -p saved_models/predictions.jsonl 
```
