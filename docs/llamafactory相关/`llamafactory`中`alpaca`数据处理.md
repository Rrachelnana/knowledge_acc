#### `llamafactory`中`alpaca`数据处理

1. `llama-factory`中`src`下的`train.py`文件开始，传入`arg `数据，这里为`examples/train_lora/llama3_lora_sft.yaml`

2. `train.py`中主要运行的是` llamafactory.train.tuner`中的`run_exp`函数，`run_exp`函数的第一步就是调用`get_train_args(agrs) `来获取模型参数、数据参数、训练参数、微调参数以及生成参数。

   1. `get_train_args` 函数中首先调用的是` _parse_train_args(args) `函数
   2. 此处采用的是 `HfArgumentParser`返回`parse`，接着为`_parse_args(parser,args)`对`yaml`文件进行解析

3. 根据`yaml`文件返回结果调用`run_sft`函数，并传入对应的参数

4. 根据 `get_dataset`函数获取`dataset`：`dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)`  

   1. 在`get_dataset`函数中，首先获取对应的`template`，获取的`qwen template`如下：

      ```
      _register_template(
          name="qwen",
          format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
          format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
          format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
          format_separator=EmptyFormatter(slots=["\n"]),
          default_system="You are a helpful assistant.",
          stop_words=["<|im_end|>"],
          replace_eos=True,
      )
      ```

   2. 对应修正后的template如下：

      ```
      {% set system_message = 'You are a helpful assistant.' %}
      {% if messages[0]['role'] == 'system' %}
          {% set system_message = messages[0]['content'] %}
      {% endif %}
      {% if system_message is defined %}
      	{{ '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}
      {% endif %}
      {% for message in messages %}
          {% set content = message['content'] %}
          {% if message['role'] == 'user' %}
              {{ '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n' }}
          {% elif message['role'] == 'assistant' %}
              {{ content + '<|im_end|>' + '\n' }}
          {% endif %}
      {% endfor %}
      ```

      

   3. 接着通过 `load_single_dataset `进行数据`align_dataset` `(llamafactory.data.aligner中的align_dataset) ` ，最后返回的`dataset.map `函数中调用`convert_alpaca`对`dataset`进行了转换加载。

      ​			此处通过`dataset.map`函数进行`convert_alpaca`函数时，需要将`kwargs`中l`oad_from_cache_file`设置为`false`，否则会从缓存中加载，而不会进入`convert_alpacha`函数中。

      1. `convert_alpaca`函数`（llamafactory.data.aligner）`传入的`examples`为加载的数据，并将其存为一个字典，其`key`为`intruction、input、output`，其`value`为列表，保存的数据。

      2. 后续按行数，将数据存入到`output`中。

         ```
         outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
         ```

         其中`prompt` 对应`intruction`，`response`为`output`，`input`为`query`。

         拼接后的`prompt`：

         ```
         prompt = []
         content = []
         if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
         	content.append(examples[dataset_attr.prompt][i])
         
         if dataset_attr.query and examples[dataset_attr.query][i]:
         	content.append(examples[dataset_attr.query][i])
         prompt.append({"role": Role.USER.value, "content": "\n".join(content)})
         
         [{'role': 'user', 'content': '请根据输入的工步名称生成对应的动素和工时，以json格式输出\n装配车轮总成到制动器（左前）'}]
         ```

         拼接后的`response`：

         ```
         if dataset_attr.response and isinstance(examples[dataset_attr.response][i], str): 
         	response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
         
         [{'role': 'assistant', 'content': '```json\n{\n    "operation_name": "放置前保险杠右气帘至前保险杠",\n    "activities": [\n        {\n            "activity": "放置前保险杠右气帘至前保险杠",\n            "elements": [\n                {\n                    "action_content": "放置零部件到安装位置",\n                    "tmu": 30\n                },\n                {\n                    "action_content": "调整对准",\n                    "tmu": 30\n                }\n            ]\n        }\n    ]\n}\n```'}]
         ```

         

      3. 最后将保存的数据为`output`

