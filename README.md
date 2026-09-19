# LLM4EEG2Text
This repository contains the code of our Expert Systems With Applications (EWSA) paper:

[Guiding LLMs to Decode Text via Aligning Semantics in EEG Signals and Language](https://www.sciencedirect.com/science/article/pii/S0957417425039156)

We built our code based on https://github.com/NeuSpeech/EEG-To-Text.


## Download ZuCo datasets
1. Download ZuCo v1.0 'Matlab files' for 'task1-SR','task2-NR','task3-TSR' from https://osf.io/q3zws/files/ and download ZuCo v2.0 'Matlab files' for 'task1-NR' from https://osf.io/2urht/files/.

2. unzip and move all `.mat` files to `~/datasets/ZuCo/task1-SR/`Matlab_files, `~/datasets/ZuCo/task2-NR/Matlab_files`, `~/datasets/ZuCo/task3-TSR/Matlab_files` and `~/datasets/ZuCo/task2-NR-2.0/Matlab_files` respectively.

## Preprocess datasets
```
bash ./scripts/prepare_dataset.sh
```


## Train Model
```
bash ./scripts/train_decoding.sh
```

## Evaluation
```
bash ./scripts/eval_decoding.sh
```

## Polymarket "Superforecaster" user search
Use `/home/runner/work/LLM4EEG2Text/LLM4EEG2Text/util/polymarket_superforecasters.py` to rank users who:
- are active at least once per week across their observed period,
- are filtered for non-robotic behavior (caps on daily/weekly volume and minimum median gap between actions),
- maximize the edge between realized win chance and market probability at entry.

Supported inputs: `.csv`, `.json`, `.jsonl`.

Expected fields (module supports aliases):
- `user_id`
- `timestamp`
- `event_id`
- `side` (`yes/no`, `1/0`, etc.)
- `entry_probability` (`0..1` or `0..100`)
- `resolved_outcome` (`yes/no`, `1/0`, etc.)

Example:
```
python /home/runner/work/LLM4EEG2Text/LLM4EEG2Text/util/polymarket_superforecasters.py \
  --input /absolute/path/to/actions.csv \
  --top-k 20 \
  --min-expected-weeks 4 \
  --max-actions-per-day 40 \
  --max-actions-per-week 150 \
  --max-avg-actions-per-week 60 \
  --min-median-gap-seconds 10 \
  --min-resolved-actions 8
```

## Citation
```
@article{Zheng2025GuidingLT,
  title={Guiding LLMs to Decode Text via Aligning Semantics in EEG Signals and Language},
  author={Huanran Zheng and Yuanbin Wu and Tianwen Qian and Wenjing Yue and Xiaoling Wang},
  journal={Expert Systems with Applications},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:282876573}
}
```