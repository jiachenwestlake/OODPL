# OODPL
Code for "[Generalizing reward modeling for out-of-distribution preference learning](https://arxiv.org/abs/2402.14760)" in ECML PKDD'2024


### Introduction
The code is built using the open-source toolkit [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). 

### Requirements
```
torch>=1.13.1
transformers>=4.31.0
datasets>=2.12.0
accelerate>=0.21.0
peft>=0.4.0
trl>=0.7.1
```

### Usage
**Dataset**  
The datasets include the Stanford Human Preferences Dataset [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)   
The dataset can be downloaded at [Dir](https://pan.baidu.com/s/1wcSm1hGxf13gbSu_Nk4OsA?pwd=jiac).

**Training command**
```
accelerate launch --config_file accelerate_config.yaml src/train_bash.py \
    --stage cpl \
    --cpl_config prob \
    --cl_alpha 0.1 \
    --reg_alpha 0.1 \
    --model_name_or_path <model path> \
    --do_train \
    --dataset <dataset path> \
    --cpl_split pairwise \
    --mix_strategy interleave_multitask \
    --template default \
    --finetuning_type full \
    --output_dir <output directory> \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 3e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --bf16
```

**Prediction command**
```
CUDA_VISIBLE_DEVICES=5 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path <trained model path> \
    --dataset shdx  \
    --template default \
    --finetuning_type full \
    --output_dir <output directory> \
    --per_device_eval_batch_size 1 \
    --max_samples 1000 \
    --predict_with_generate \
    --bf16
```

### Citation
```
@inproceedings{jia2024generalizing,
  title={Generalizing Reward Modeling for Out-of-Distribution Preference Learning},
  author={Jia, Chen},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={107--124},
  year={2024},
  organization={Springer}
}
```
