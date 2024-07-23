##### 解析命令行 `lamafactory-cli train`

```
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

1. 由`setup.py`文件知道  `llamafactory-cli` 命令等价于执行 `llamafactory.cli:main`



2. `cli.py` 文件中的`main`函数 ，通过`sys.argv.pop(1)` 提取上述命令中的 `train`

```
command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
```



3. 接着通过 `torchrun` 进行运行

```
"torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
 "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
```

具体命令如下：

```
if command == Command.TRAIN:
        force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
        if force_torchrun or get_device_count() > 1:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                ).format(
                    nnodes=os.environ.get("NNODES", "1"),
                    node_rank=os.environ.get("RANK", "0"),
                    nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                ),
                shell=True,
            )
            sys.exit(process.returncode)
        else:
            run_exp()
```



4. `file_name` 为 `launcher.__file__` ， 该文件的参数为`args` ，为`sys.argv[1:]`

```
from . import launcher
```



5. 当前目录的 `launcher.py` 文件 仅仅执行了 `run_exp`函数

   ```
   from llamafactory.train.tuner import run_exp
   
   def launch():
       run_exp()
   
   if __name__ == "__main__":
       launch()
   ```

   

   6. 在`tuner.py`文件中， 导入的是`run_sft` 函数

      ```
      from .sft import run_sft
      ```

6. 在`sft` 的`__init__.py`文件中，看到执行的是当前目录的`workflow.py`中的 `run_sft` 函数

   ```
   from .workflow import run_sft
   __all__ = ["run_sft"]
   ```



7. `llama3_lora_sft.yaml` 文件

   ```
   ### model
   #指定模型的路径或名称
   model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct 
   
   ### method
   stage: sft             # 设置为pt，表示预训练阶段
   do_train: true         # 设置为true，表示执行训练
   finetuning_type: lora  # 设置为lora，表示使用LoRA微调
   lora_target: all       # 设置为all，表示所有模块都应用LoRA
   
   ### dataset
   dataset: identity,alpaca_en_demo
   template: llama3
   cutoff_len: 1024                    # 设置为1024，表示最大序列长度
   max_samples: 1000                   # 设置为1000，表示最大样本数
   overwrite_cache: true               # 设置为true，表示为覆盖缓存
   preprocessing_num_workers: 16       # 设置为16，表示使用16个工作线程进行预处理
   
   ### output
   output_dir: saves/llama3-8b/lora/sft # 指定输出目录
   logging_steps: 10                    # 设置为10，表示每10步记录一次日志
   save_steps: 500                      # 设置为500，表示每500步保存一次模型
   plot_loss: true                      # 设置为true，表示绘制损失图
   overwrite_output_dir: true           # 设置为true，表示覆盖输出目录
   
   ### train
   per_device_train_batch_size: 1       # 设置为 1，表示每个设备的训练批大小为 1
   gradient_accumulation_steps: 8       # 设置为 8，表示梯度累积步数为 8，即每 8 个批次更新一次梯度
   learning_rate: 1.0e-4                # 设置为 1.0e-4，表示学习率为 0.0001
   num_train_epochs: 3.0                # 设置为 3.0，表示训练 3 个 epoch
   lr_scheduler_type: cosine            # 设置为 cosine，表示使用余弦退火学习率调度器
   warmup_ratio: 0.1                    # 设置为 0.1，表示学习率预热比例为 10%
   bf16: true                           # 设置为 true，表示使用 bfloat16 精度
   ddp_timeout: 180000000               # 设置为 180000000，表示分布式数据并行超时时间
   
   ### eval
   val_size: 0.1                        # 设置为 0.1，表示验证集占数据集的 10%
   per_device_eval_batch_size: 1        # 设置为 1，表示每个设备的验证批大小为 1
   eval_strategy: steps                 # 设置为 steps，表示按步数进行评估
   eval_steps: 500                      # e设置为 500，表示每 500 步进行一次评估 
   ```

   

使用 `.sh` 脚本传参虽然直观，但在参数较多时容易变得杂乱且难以管理。`yaml` 文件更适合复杂配置的场景，特别是在需要频繁调整和复用配置的情况下

`.sh` 脚本传参方式

在`.sh`脚本中，参数是直接通过命令行传递的。 

```
accelerate launch src/train_bash.py \
    --model_name_or_path /data/vayu/train/etuning/LLaMA-Factory/models/xxxx-Base-10B-200k-Llama \
    --do_train \
    --stage pt
    # ... 更多参数
```

每个参数都需要以 --参数名 参数值的形式显示地列出。

这种方式直观，但参数较多时会显得冗长且难易管理。



`.yaml` 文件传参方式

参数以键值对的形式组织在文件中

```
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
 
### method
stage: pt
do_train: true
finetuning_type: lora
lora_target: all
 
# ... 更多参数
```

这种方式将所有配置参数集中在一个文件中，结构更清晰，管理和修改都更加方便。



`yaml` 配置文件的优势

1. 可读性和可维护性
   1. `yaml`  文件使用层次化的结构，参数分类更清晰。
   2. 便于阅读和理解，特别是当参数较多时。
2. 版本控制
   1. 配置文件可以很方便地进行版本控制，通过比较不同版本的配置文件，可以轻松追踪参数的变化。
3. 复用性
   1. 可以将 `.yaml` 文件复用在不同的训练任务中，只需修改部分参数即可，而不需要每次都重新编写整个命令行。
4. 减少错误
   1. 通过文件传参，避免了命令行传参时可能出现的拼写错误或遗漏参数的情况。


