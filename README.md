## lora-chatglm 版本

## Finetune步骤，顺序执行即可

### 环境

- 显卡: 显存 >= 16G (最好24G或者以上)
- 环境：
- - python>=3.8
- - cuda>=11.6
- - pip install -r requirements.txt
- - 修改模型路径为本地路径

### 1、构建数据集
执行 data_process文件内的 trans_doc2json.ipynb文件
构建data 文件下的 military 数据集


### 2、数据处理

 - 转化data里的military数据集为jsonl形式

```bash
python cover2jsonl.py \
    --data_path data/military.json \
    --save_path data/military.jsonl
```

 - tokenization

```bash
python tokenize_dataset_rows.py \
    --jsonl_path data/military.jsonl \
    --save_path data/military \
    --max_seq_length 200 \ 
    --skip_overlength
```

- `--jsonl_path` 微调的数据路径, 格式jsonl, 对每行的['context']和['target']字段进行encode
- `--save_path` 输出路径
- `--max_seq_length` 样本的最大长度

###  3、训练

```bash
python finetune.py \
    --dataset_path data/military \
    --lora_rank 32 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
```

### 4、使用预训练好的LoRA

参考 [examples/infer_pretrain.ipynb]路径# dplus-lora
