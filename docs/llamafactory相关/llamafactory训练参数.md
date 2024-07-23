#### llamafactory训练参数

llamafactory.hparams.parser中get_train_args函数中：

```python
# 检查training_args对象中的parallel_mode属性是否被设置为ParallelMode.NOT_DISTRIBUTED
# ParallelMode可能是一个枚举类型，其中NOT_DISTRIBUTED表示非分布式训练模式。
if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
    # 训练参数中设置的并行模式为非分布式训练，则抛出一个ValueError异常。异常消息提示用户应该使用llamafactory-cli或torchrun来启动分布式训练。
	raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")
```

​		该代码目的是确保训练过程是在分布式模式下进行的。如果用户尝试在没有设置为分布式训练模式的情况下启动训练，代码将抛出异常并提示用户如何正确启动分布式训练。



vscode 指定用某张卡进行模型训练，在debug.configurations进行设置

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {	
        	# 提供 VSCode 下拉列表中显示的调试配置的名称
            "name": "Python Debugger: Current File",
            # 标识要使用的调试器类型
            "type": "debugpy",
            # 指定开始调试的模式：
            # launch：在指定的文件上启动调试器program
            # attach：将调试器附加到已经运行的进程
            "request": "launch",
            # 提供 python 程序的入口模块（启动文件）的完全限定路径
            "program": "src/train.py",
            # VS Code 调试控制台
            # redirectOutput指定在不修改默认值的情况下程序输出的显示方式。
            "console": "integratedTerminal",
            # 指定要传递给 Python 程序的参数。由空格分隔的参数字符串的每个元素都应包含在引号内
            "args": ["examples/train_lora/llama3_lora_sft.yaml"],
            "env": {
                "CUDA_VISIBLE_DEVICES": "1"
            },
    		# 使用语法指定要传递给 Python 解释器的参数"pythonArgs": ["<arg 1>", "<arg 2>",...]
    		"pythonArgs":[],
    		# 指向用于调试的 Python 解释器的完整路径
    		”python“：”“，
        }
    ]
}
```

更多参数指令函数参考：[vscode中的python调试](https://vscode.github.net.cn/docs/python/debugging)





参考文献：

[VSCode 中的Python调试](https://vscode.github.net.cn/docs/python/debugging)

