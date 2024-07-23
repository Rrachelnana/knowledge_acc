torchrun相关命令

```
torchrun --standalone --nproc_per_node=gpu XXX.py
```

1. `--standalone` 代表单机运行
2. `--nproc_per_node=gpu` 代表使用所有可用GPU。等于号后也可写gpu数量n，这样会使用前n个GPU



