python基础知识点

1、在字符串中使用变量

```python
first_name='ada'
last_name='lovelace'
full_name=f'{first_name} {last_name}' #两个花括号中需空一格
```

f 是3.6版本引入的，如果是3.5及以下版本需要使用format函数

2、制表符 `\t`  换行符 `\n`

3. rstrip 剔除右边空白  lstrip  剔除左边空白   strip剔除左右空白
4. str.title  str字符串以大写字符开始

5.  列表末尾添加元素 哟个 -append
6. 列表插入元素 -insert   list1.insert(0,'ll')
7. 列表删除元素  del  del list1[0]
8. 列表中删除元素  pop(位置索引)
9. 列表 删除元素  remove
10. 列表排序  list1.sort()  永久的     sorted(list1) 临时的
11. list1.reverse()反转列表元素    永久的



##### os

os.path.isdir()函数判断某一路径是否为目录

```
os.path.isdir(path) 
```

| 方法                                | 说明                                                         |
| ----------------------------------- | :----------------------------------------------------------- |
| os.path.abspath(path)               | 返回 path 的绝对路径。                                       |
| os.path.basename(path)              | 获取 path 路径的基本名称，即 path 末尾到最后一个斜杠的位置之间的字符串。 |
| os.path.commonprefix(list)          | 返回 list（多个路径）中，所有 path 共有的最长的路径。        |
| os.path.dirname(path)               | 返回 path 路径中的目录部分。                                 |
| os.path.exists(path)                | 判断 path 对应的文件是否存在，如果存在，返回 True；反之，返回 False。和 lexists() 的区别在于，exists()会自动判断失效的文件链接（类似 Windows 系统中文件的快捷方式），而 lexists() 却不会。 |
| os.path.lexists(path)               | 判断路径是否存在，如果存在，则返回 True；反之，返回 False。  |
| os.path.expanduser(path)            | 把 path 中包含的 "~" 和 "~user" 转换成用户目录。             |
| os.path.expandvars(path)            | 根据环境变量的值替换 path 中包含的 "$name" 和 "${name}"。    |
| os.path.getatime(path)              | 返回 path 所指文件的最近访问时间（浮点型秒数）。             |
| os.path.getmtime(path)              | 返回文件的最近修改时间（单位为秒）。                         |
| os.path.getctime(path)              | 返回文件的创建时间（单位为秒，自 1970 年 1 月 1 日起（又称 Unix 时间））。 |
| os.path.getsize(path)               | 返回文件大小，如果文件不存在就返回错误。                     |
| os.path.isabs(path)                 | 判断是否为绝对路径。                                         |
| os.path.isfile(path)                | 判断路径是否为文件。                                         |
| os.path.isdir(path)                 | 判断路径是否为目录。                                         |
| os.path.islink(path)                | 判断路径是否为链接文件（类似 Windows 系统中的快捷方式）。    |
| os.path.ismount(path)               | 判断路径是否为挂载点。                                       |
| os.path.join(path1[, path2[, ...]]) | 把目录和文件名合成一个路径。                                 |
| os.path.normcase(path)              | 转换 path 的大小写和斜杠。                                   |
| os.path.normpath(path)              | 规范 path 字符串形式。                                       |
| os.path.realpath(path)              | 返回 path 的真实路径。                                       |
| os.path.relpath(path[, start])      | 从 start 开始计算相对路径。                                  |
| os.path.samefile(path1, path2)      | 判断目录或文件是否相同。                                     |
| os.path.sameopenfile(fp1, fp2)      | 判断 fp1 和 fp2 是否指向同一文件。                           |
| os.path.samestat(stat1, stat2)      | 判断 stat1 和 stat2 是否指向同一个文件。                     |
| os.path.split(path)                 | 把路径分割成 dirname 和 basename，返回一个元组。             |
| os.path.splitdrive(path)            | 一般用在 windows 下，返回驱动器名和路径组成的元组。          |
| os.path.splitext(path)              | 分割路径，返回路径名和文件扩展名的元组。                     |
| os.path.splitunc(path)              | 把路径分割为加载点与文件。                                   |
| os.path.walk(path, visit, arg)      | 遍历path，进入每个目录都调用 visit 函数，visit 函数必须有 3 个参数(arg, dirname, names)，dirname 表示当前目录的目录名，names 代表当前目录下的所有文件名，args 则为 walk 的第三个参数。 |
| os.path.supports_unicode_filenames  | 设置是否可以将任意 Unicode 字符串用作文件名。                |

##### shutil模块详解

os模块是Python标准库中一个重要的模块，里面提供了对目录和文件的一般常用操作。而Python另外一个标准库——shutil库，它作为os模块的补充，提供了复制、移动、删除、压缩、解压等操作，这些 os 模块中一般是没有提供的。但是需要注意的是：shutil 模块对压缩包的处理是调用 ZipFile 和 TarFile这两个模块来进行的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020081717024675.png)

[shutil](https://so.csdn.net/so/search?q=shutil&spm=1001.2101.3001.7020).rmtree() 表示递归删除文件夹下的所有子文件夹和子文件。



##### distutils 

Distutils可以用来在Python环境中构建和安装额外的模块。新的模块可以是纯Python的，也可以是用C/C++写的扩展模块，或者可以是Python包，包中包含了由C和Python编写的模块。



##### dockerfile

1.写一个Dockerfile文件。把这个文件放在项目的目录中，这个文件中包含了一条条的指令(Instruction)，每一条指令构建一层，因此每一条指令的内容，就是描述该层应当如何构建。

2.使用`docker build 镜像名称 上下文路径`的方式构建镜像

###### **常用的dockerfile指令以及编写注意事项**

1. FROM 需要修改的镜像名称
2. MAINTAINER 标明作者
3. ARG： 定义构建过程中需要的参数
4. ENV： 在镜像的构建过程中设置环境变量
5. EXPOSE ：暴露端口，非必须
6. WORKDIR： 设置工作目录
7. ADD&&COPY:同样是将本地的文件/目录拷贝到docker中

区别在于copy不会对压缩文件进行解压，也不能通过url链接进行拷贝。一般的语法为`ADD [本地路径/文件] [虚拟环境路径]`

	8. RUN： 运行指令，比如apt-get install一些软件啥的
	9. ENTRYPOINT&&CMD：都是在docker镜像启动时，执行命令。



##### python后缀

| 后缀名 | 作用                                                         |
| :----- | :----------------------------------------------------------- |
| py     | 最常见的 Python 源代码文件。                                 |
| pyc    | 常见的 Python 字节码缓存文件，可以反编译成 py 文件。         |
| pyo    | 另一种 Python 字节码缓存文件，只存在于 Python2 及 Python3.5 之前的版本。 |
| pyi    | Python 的存根文件，常用于 IDE 代码格式检查时的类型提示。     |
| pyw    | 另一种 Python 源代码文件，一般只存在于 Windows 系统。        |
| pyd    | 一种 Python 可直接调用的 C 语言动态链接库文件，一般只存在于 Windows 系统。 |
| pyx    | Cython 源代码文件，一般用来编写 Python 的 C 扩展。           |

pyx    Cython 源代码文件。

注意是 Cython 不是 CPython。Cython 可以说是一种编程语言， 它结合了Python 的语法和有 C/C++的效率，用 Cython 写完的代码可以很容易转成 C 语言代码，然后又可以再编译成动态链接库(`pyd`或`dll`)供 Python 调用，所以 Cython 一般用来编写 Python 的 C 扩展，上面说的 Python 文件编译生成 `pyd` 文件就是利用 Cython 来实现的 。Cython 的源代码文件一般为`pyx`后缀。



##### inverse_transform

X1=scaler.inverse_transform(X_scaled)是将标准化后的数据转换为原始数据。



##### onnxruntime

ONNX （Open Neural Network Exchange）是 Facebook 和微软在2017年共同发布的，用于标准描述计算图的一种格式。目前，在数家机构的共同维护下，ONNX 已经对接了多种深度学习框架和多种推理引擎。因此，ONNX 被当成了深度学习框架到推理引擎的桥梁，就像编译器的中间语言一样。由于各框架兼容性不一，我们通常只用 ONNX 表示更容易部署的静态图。

```
onnxruntime  高性能，跨平台的深度学习推理引擎ONNX Runtime
```

- 最大化自动地在不同的平台上利用定制的accelerators和runtimes。
- 流程非常简单。 从ONNX模型开始，ONNXRuntime首先将模型图转换为其内存中的图表示形式。 然后，它应用许多graph transformations，这些转换包括：a）执行一组独立于提供程序的优化，例如float16和float32之间的转换转换，以及b）根据可用的execution provider将图形划分为一组子图。 每个子图都分配给一个execution provider。 通过使用GetCapability（）API查询execution provider的功能，使得我们确保可以由对应的execution provider执行子图。



![image-20220505092823394](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220505092823394.png)



- ONNXRuntime根据可用的execution provider将模型图划分为子图，每个execution provider一个子图。 



##### autokeras

```
autokeras  自动化机器学习，通常被称为AutoML，是自动化构建指神经网络结构
```

AutoML通过智能架构的操作，可以让大家更方便、更快速的进行深度学习的研究。

通过AutoKeras这个神经架构搜索算法，我们可以找到最好的神经网络架构，比如层中神经元的数量，架构的层数，加入哪些层，层的特定参数，比如Dropout中的滤波器大小或掉落神经元的百分比等等。当搜索完成后，你可以将模型作为常规的TensorFlow/Keras模型使用。

[StructuredDataRegressor - AutoKeras](https://autokeras.com/structured_data_regressor/)

```
autokeras.StructuredDataRegressor(
    column_names=None,
    column_types=None,
    output_dim=None,
    loss="mean_squared_error",
    metrics=None,
    project_name="structured_data_regressor",
    max_trials=100,
    directory=None,
    objective="val_loss",
    tuner=None,
    overwrite=False,
    seed=None,
    max_model_size=None,
    **kwargs
)
```



```
StructuredDataRegressor.fit(
    x=None, y=None, epochs=None, callbacks=None, validation_split=0.2, validation_data=None, **kwargs
)
```

Search for the best model and hyperparameters for the AutoModel.

```
StructuredDataRegressor.predict(x, **kwargs)
```

Predict the output for a given testing data.

```
StructuredDataRegressor.evaluate(x, y=None, **kwargs)
```

Evaluate the best model for the given data.

```
StructuredDataRegressor.export_model()
```

Export the best Keras Model.

```
model = clf.export_model()
# 保存模型
try:
    model.save("model_autokeras", save_format="tf")
except:
    model.save("model_autokeras.h5")
# 导入模型
from tensorflow.keras.models import load_model
loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
# 预测数据
predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
print(predicted_y)

```





##### **wandb**

**wandb(Weights & Biases)**是一个类似于tensorboard的极度丝滑的在线模型训练可视化工具。

**wandb**这个库可以帮助我们**跟踪实验，记录运行中的超参数和输出指标，可视化结果并共享结果**。

下图展示了wandb这个库的功能，Framework Agnostic的意思是无所谓你用什么框架，均可使用wandb。wandb可与用户的机器学习基础架构配合使用：AWS，GCP，Kubernetes，Azure和本地机器。

![image-20220505094714754](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220505094714754.png)

wandb的重要的工具：

- Dashboard：跟踪实验，可视化结果；

- Reports：分享，保存结果；

- Sweeps：超参调优；

- Artifacts：数据集和模型的版本控制。



##### sample

[Pandas](https://so.csdn.net/so/search?q=Pandas&spm=1001.2101.3001.7020) sample()用于从DataFrame中随机选择行和列。

```
DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
```



eval

eval() 和 exec() 函数都属于 [Python](http://c.biancheng.net/python/) 的内置函数，都可以执行一个字符串形式的 Python 代码（代码以字符串的形式提供），相当于一个 Python 的解释器。二者不同之处在于，eval() 执行完要返回结果，而 exec() 执行完不返回结果。



##### iterrows

使用iterrows()对dataframe进行遍历

```
import pandas as pd

otu = pd.read_csv("otu.txt",sep="\t")
for index,row in otu.iterrows():
  print(index)
  print(row)
```

iterrows()返回值为[元组](https://so.csdn.net/so/search?q=元组&spm=1001.2101.3001.7020),(index,row)
上面的代码里，for循环定义了两个变量，index，row，那么返回的元组，index=index，row=row.



##### conda 与pip的区别与联系

联系：conda与pip都是包管理器，有一些重叠功能

区别：conda是一个跨平台的包和环境管理器，安装的包是二进制的，同时安装包不限于python包，还可以是c、c++库、R库以及其他软件；即python安装python包，conda可安装用任何语言编写的软件包；

使用pip之前必须先安装python解释器；而conda可直接安装python包和python解释器

conda能创建独立环境，可包含不同版本的python和包，而pip不支持内置独立环境，必须依赖其他工具，如virtual来创建隔离环境

conda和pip在实现环境依赖时也不同，pip不能确保所有包的依赖关系同时满足，而conda使用satisfiability (SAT) 来检验包的所有依赖是否满足

##### zfill

```
s.zfill(width)
# width -- 指定字符串的长度。原字符串右对齐，前⾯填充0。
```



##### logging

logging.handlers.TimedRotatingFileHandler

Python 3 中的logging 日志文件的回滚模块RotatingFileHandler，主要有两种方式，一种是基于文件的大小进行回滚，第二种是基于当前系统时间进行回滚。

基于时间回滚的方法即 通过调用TimedRotatingFileHandler（）函数进行配置即可

```
handlers[level] = TimedRotatingFileHandler(path, backupCount=0, encoding=‘utf-8’,when=‘D’, interval=1)
# backupCount：允许存储的文件个数，如果大于这个数系统会删除最早的日志文件
#when=‘D’ 代表以天为间隔 S：秒 H：小时
# interval=1 代表间隔多少个 when 进行一次日志分割
```



##### 虚拟环境其他

anaconda3 清华源镜像：[Index of /anaconda/archive/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh
```



在linux系统上安装好anaconda3后，无法修改环境变量，找到conda路径

![image-20220507145905413](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220507145905413.png)

查看conda虚拟环境

```
./conda env list
```

![image-20220507150100344](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220507150100344.png)

创建conda虚拟环境

```
./conda create -n welding-parameter python=3.7.11
```

![image-20220507150631626](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220507150631626.png)



激活虚拟环境

```
source activate welding-parameter
```

![image-20220507154018365](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220507154018365.png)



首先，进入requirements.txt文件所在目录。
之后，使用命令pip install -r requirements.txt

退出虚拟环境：conda deactivate



```text
方法一：
# 第一步：首先退出环境
conda deactivate
 
# 第二步：查看虚拟环境列表，此时出现列表的同时还会显示其所在路径
conda env list
 
# 第三步：删除环境
conda env remove -p 要删除的虚拟环境路径
conda env remove -p /home/kuucoss/anaconda3/envs/tfpy36   #我的例子

方法2
# 第一步：首先退出环境
conda deactivate
 
# 第二步：删除环境
conda remove -n  需要删除的环境名 --all
```

conda环境

[(51 封私信 / 80 条消息) conda环境安装 - 收藏夹 - 知乎 (zhihu.com)](https://www.zhihu.com/collection/749707738)

```
#清华源
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
 #中科大源
conda config --remove channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/  
conda config --remove channels https://mirrors.ustc.edu.cn/anaconda/cloud/

conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/
#阿里源
conda config --add channels https://mirrors.aliyun.com/pypi/simple/               
```



```
[tangnn@cqi145-daas-gpu03 bin]$ conda config --add channels http://mirrors.aliyun.com/anaconda/pkgs/r
[tangnn@cqi145-daas-gpu03 bin]$ conda config --add channels http://mirrors.aliyun.com/anaconda/pkgs/msys2
[tangnn@cqi145-daas-gpu03 bin]$ conda config --set show_channel_urls yes
[tangnn@cqi145-daas-gpu03 bin]$ conda config --show
add_anaconda_token: True
add_pip_as_python_dependency: True
aggressive_update_packages:
  - ca-certificates
  - certifi
  - openssl
allow_conda_downgrades: False
allow_cycles: True
allow_non_channel_urls: False
allow_softlinks: False
always_copy: False
always_softlink: False
always_yes: None
anaconda_upload: None
auto_activate_base: True
auto_stack: 0
auto_update_conda: True
bld_path: 
changeps1: True
channel_alias: https://conda.anaconda.org
channel_priority: flexible
channels:
  - http://mirrors.aliyun.com/anaconda/pkgs/msys2
  - http://mirrors.aliyun.com/anaconda/pkgs/r
  - http://mirrors.aliyun.com/anaconda/pkgs/main
  - defaults
client_ssl_cert: None
client_ssl_cert_key: None
clobber: False
conda_build: {}
create_default_packages: []
croot: /home/tangnn/anaconda3/conda-bld
custom_channels:
  pkgs/main: https://repo.anaconda.com
  pkgs/r: https://repo.anaconda.com
  pkgs/pro: https://repo.anaconda.com
custom_multichannels:
  defaults: 
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
  local: 
debug: False
default_channels:
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
default_python: 3.8
default_threads: None
deps_modifier: not_set
dev: False
disallowed_packages: []
download_only: False
dry_run: False
enable_private_envs: False
env_prompt: ({default_env}) 
envs_dirs:
  - /home/tangnn/anaconda3/envs
  - /home/tangnn/.conda/envs
error_upload_url: https://conda.io/conda-post/unexpected-error
execute_threads: 1
extra_safety_checks: False
force: False
force_32bit: False
force_reinstall: False
force_remove: False
ignore_pinned: False
json: False
local_repodata_ttl: 1
migrated_channel_aliases: []
migrated_custom_channels: {}
non_admin_enabled: True
notify_outdated_conda: True
offline: False
override_channels_enabled: True
path_conflict: clobber
pinned_packages: []
pip_interop_enabled: False
pkgs_dirs:
  - /home/tangnn/anaconda3/pkgs
  - /home/tangnn/.conda/pkgs
proxy_servers: {}
quiet: False
remote_backoff_factor: 1
remote_connect_timeout_secs: 9.15
remote_max_retries: 3
remote_read_timeout_secs: 60.0
repodata_fns:
  - current_repodata.json
  - repodata.json
repodata_threads: None
report_errors: None
restore_free_channel: False
rollback_enabled: True
root_prefix: /home/tangnn/anaconda3
safety_checks: warn
sat_solver: pycosat
separate_format_cache: False
shortcuts: True
show_channel_urls: True
signing_metadata_url_base: https://repo.anaconda.com/pkgs/main
solver_ignore_timestamps: False
ssl_verify: True
subdir: linux-64
subdirs:
  - linux-64
  - noarch
target_prefix_override: 
track_features: []
unsatisfiable_hints: True
unsatisfiable_hints_check_depth: 2
update_modifier: update_specs
use_index_cache: False
use_local: False
use_only_tar_bz2: False
verbosity: 0
verify_threads: 1
whitelist_channels: []
```



```
pip install gunicorn -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gunicorn
```

```
[tangnn@cqi145-daas-gpu03 bin]$ pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
Writing to /home/tangnn/.config/pip/pip.conf
[tangnn@cqi145-daas-gpu03 bin]$ pip config set install.trusted-host mirrors.aliyun.com
Writing to /home/tangnn/.config/pip/pip.conf
```

设置pip，给pip添加镜像源，然后用pip install  包 进行安装



##### 问题1

```
Can't get remote credentials for deployment server tangnn@10.113.75.134:22 password
```

删除多余的解释器，仅留一个



##### 问题2

![image-20220510105654967](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20220510105654967.png)

python -m pip install dask distributed --upgrade



##### 问题3

```
AttributeError: Can't get attribute '_unpickle_timestamp' on <module 'pandas._libs.tslibs.timestamps' from '/home/tangnn/anaconda3/lib/python3.8/site-packages/pandas/_libs/tslibs/timestamps.cpython-38-x86_64-linux-gnu.so'>
```

依赖包问题



##### AES 加密

[Python AES Encryption Example - DevRescue](https://devrescue.com/simple-python-aes-encryption-example/)



##### 问题4：py文件编译的pyx文件无某变量

```
2022-08-15 11:09:45.402706: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Traceback (most recent call last):
  File "/home/tangnn/anaconda3/envs/weld-paras/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-5c1e792e2b62>", line 1, in <module>
    runfile('/home/tangnn/project/welding-parameter-prediction/trains/train_3_layers_ak.py', wdir='/home/tangnn/project/welding-parameter-prediction/trains')
  File "/home/tangnn/.pycharm_helpers/pydev/_pydev_bundle/pydev_umd.py", line 197, in runfile
    pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
  File "/home/tangnn/.pycharm_helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/home/tangnn/project/welding-parameter-prediction/trains/train_3_layers_ak.py", line 3, in <module>
    from utils import analysis_predict_results
  File "analysis_predict_results.pyx", line 6, in init analysis_predict_results
ImportError: cannot import name scaler_y
```

py文件编译问题

解决方法：笨方法，直接新建了新项目，运行该文件即可



##### 问题5：RuntimeError: Too many failed attempts to build model

```
ssh://tangnn@10.113.75.134:22/home/tangnn/anaconda3/envs/welding-parameter/bin/python3.7 -u /home/tangnn/project/welding-paras-evaluate/trains/train_2_layers_ak.py

Invalid model 5/5
Traceback (most recent call last):
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/keras_tuner/engine/tuner.py", line 158, in _try_build
    model = self._build_hypermodel(hp)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/keras_tuner/engine/tuner.py", line 146, in _build_hypermodel
    model = self.hypermodel.build(hp)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/keras_tuner/engine/hypermodel.py", line 111, in _build_wrapper
    return self._build(hp, *args, **kwargs)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/graph.py", line 250, in build
    outputs = block.build(hp, inputs=temp_inputs)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/engine/block.py", line 38, in _build_wrapper
    return super()._build_wrapper(hp, *args, **kwargs)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/keras_tuner/engine/hypermodel.py", line 111, in _build_wrapper
    return self._build(hp, *args, **kwargs)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/blocks/wrapper.py", line 249, in build
    output_node = block.build(hp, output_node)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/engine/block.py", line 38, in _build_wrapper
    return super()._build_wrapper(hp, *args, **kwargs)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/keras_tuner/engine/hypermodel.py", line 111, in _build_wrapper
    return self._build(hp, *args, **kwargs)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/blocks/preprocessing.py", line 313, in build
    return keras_layers.MultiCategoryEncoding(encoding)(input_node)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/keras_layers.py", line 86, in __init__
    self.encoding_layers.append(layers.StringLookup())
AttributeError: module 'tensorflow.keras.layers' has no attribute 'StringLookup'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/tangnn/project/welding-paras-evaluate/trains/train_2_layers_ak.py", line 29, in <module>
    his = reg.fit(datasets.X_train, datasets.y_train, validation_split=0.2, epochs=epochs, verbose=1)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/tasks/structured_data.py", line 146, in fit
    **kwargs
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/auto_model.py", line 299, in fit
    **kwargs
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/autokeras/engine/tuner.py", line 191, in search
    self._try_build(hp)
  File "/home/tangnn/anaconda3/envs/welding-parameter/lib/python3.7/site-packages/keras_tuner/engine/tuner.py", line 166, in _try_build
    raise RuntimeError("Too many failed attempts to build model.")
RuntimeError: Too many failed attempts to build model.
```

conda环境中的tensorflow与autokeras版本不兼容

解决方法：虚拟环境中查看conda list |grep tensorflow ; pip list | grep tensorflow ，两版本不一致，直接卸载conda里的tensorflow：conda uninstall tensorflow 然而卸载不成功。

最终解决方法，从虚拟环境的site-packages（路径：anaconda3/envs/welding-parameter/lib/python3.7/site-packages）直接删除tensorflow相关文件，然后conda install tensorflow，再pip install tensorflow==2.9.1



##### 问题6

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ pip install -r requirements.txt
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
Collecting git+https://github.com/PanQiWei/AutoGPTQ.git (from -r requirements.txt (line 4))
  Cloning https://github.com/PanQiWei/AutoGPTQ.git to /tmp/pip-req-build-a6i8fopk
  Running command git clone --quiet https://github.com/PanQiWei/AutoGPTQ.git /tmp/pip-req-build-a6i8fopk
  Resolved https://github.com/PanQiWei/AutoGPTQ.git to commit d2662b18bb91e1864b29e4e05862712382b8a076
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      Building cuda extension requires PyTorch (>=1.13.0) being installed, please install PyTorch first: No module named 'torch'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

将setuptools包删除

输入命令：

```undefined
pip uninstall setuptools
```



问题7

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ pip install -r requirements.txt
Error processing line 1 of /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/distutils-precedence.pth:

  Traceback (most recent call last):
    File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site.py", line 186, in addpackage
      exec(line)
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named '_distutils_hack'

Remainder of file ignored
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
Collecting git+https://github.com/PanQiWei/AutoGPTQ.git (from -r requirements.txt (line 4))
  Cloning https://github.com/PanQiWei/AutoGPTQ.git to /tmp/pip-req-build-7qlmibn0
  Running command git clone --quiet https://github.com/PanQiWei/AutoGPTQ.git /tmp/pip-req-build-7qlmibn0
  fatal: unable to access 'https://github.com/PanQiWei/AutoGPTQ.git/': Failed connect to github.com:443; Connection timed out
  error: subprocess-exited-with-error
  
  × git clone --quiet https://github.com/PanQiWei/AutoGPTQ.git /tmp/pip-req-build-7qlmibn0 did not run successfully.
  │ exit code: 128
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× git clone --quiet https://github.com/PanQiWei/AutoGPTQ.git /tmp/pip-req-build-7qlmibn0 did not run successfully.
│ exit code: 128
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

如果直接使用下面安装命令

```
pip install git+https://github.com/cocodataset/panopticapi.git
```

会出现安装失败的提示的情况时，可以尝试如下方式：
将命令进行分解

```
git clone https://github.com/cocodataset/panopticapi.git
cd panopticapi
pip install -e .                          
```



```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ git clone https://github.com/PanQiWei/AutoGPTQ.git
Cloning into 'AutoGPTQ'...
remote: Enumerating objects: 3965, done.
remote: Counting objects: 100% (117/117), done.
remote: Compressing objects: 100% (75/75), done.
remote: Total 3965 (delta 71), reused 78 (delta 42), pack-reused 3848
Receiving objects: 100% (3965/3965), 7.93 MiB | 5.47 MiB/s, done.
Resolving deltas: 100% (2538/2538), done.
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ cd AutoGPTQ
(llmmodel) [tangnn@cqi145-daas-gpu03 AutoGPTQ] (main)$ pip install -e .
Error processing line 1 of /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/distutils-precedence.pth:

  Traceback (most recent call last):
    File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site.py", line 186, in addpackage
      exec(line)
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named '_distutils_hack'

Remainder of file ignored
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
Obtaining file:///data/tangnn/other_projects/Llama2-Chinese-main/AutoGPTQ
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build editable did not run successfully.
  │ exit code: 1
  ╰─> [19 lines of output]
      Error processing line 1 of /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/distutils-precedence.pth:
      
        Traceback (most recent call last):
          File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site.py", line 186, in addpackage
            exec(line)
          File "<string>", line 1, in <module>
        ModuleNotFoundError: No module named '_distutils_hack'
      
      Remainder of file ignored
      Error processing line 1 of /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/distutils-precedence.pth:
      
        Traceback (most recent call last):
          File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site.py", line 186, in addpackage
            exec(line)
          File "<string>", line 1, in <module>
        ModuleNotFoundError: No module named '_distutils_hack'
      
      Remainder of file ignored
      Building cuda extension requires PyTorch (>=1.13.0) being installed, please install PyTorch first: No module named 'torch'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build editable did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

> pip install setuptools==57.5.0

执行成功

> pip install easydict

然后再安装AutoGPTQ

```
(llmmodel) [tangnn@cqi145-daas-gpu03 AutoGPTQ] (main)$ pip install -e .
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
Obtaining file:///data/tangnn/other_projects/Llama2-Chinese-main/AutoGPTQ
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      Building cuda extension requires PyTorch (>=1.13.0) being installed, please install PyTorch first: No module named 'torch'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

```

安装requirements上要求的torch

```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

再次安装AutoGPTQ，报如下错误

```
(llmmodel) [tangnn@cqi145-daas-gpu03 AutoGPTQ] (main)$ pip install -e .
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
Obtaining file:///data/tangnn/other_projects/Llama2-Chinese-main/AutoGPTQ
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [22 lines of output]
      Traceback (most recent call last):
        File "/data/tangnn/other_projects/Llama2-Chinese-main/AutoGPTQ/./autogptq_extension/qigen/generate.py", line 8, in <module>
          from gekko import GEKKO
      ModuleNotFoundError: No module named 'gekko'
      Traceback (most recent call last):
        File "/data/tangnn/other_projects/Llama2-Chinese-main/AutoGPTQ/setup.py", line 109, in <module>
          subprocess.check_output(["python", "./autogptq_extension/qigen/generate.py", "--module", "--search", "--p", str(p)])
        File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/subprocess.py", line 421, in check_output
          return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
        File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/subprocess.py", line 526, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['python', './autogptq_extension/qigen/generate.py', '--module', '--search', '--p', '24']' returned non-zero exit status 1.
      
      During handling of the above exception, another exception occurred:
      
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/data/tangnn/other_projects/Llama2-Chinese-main/AutoGPTQ/setup.py", line 111, in <module>
          raise Exception(f"Generating QiGen kernels failed with the error shown above.")
      Exception: Generating QiGen kernels failed with the error shown above.
      Generating qigen kernels...
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

安装gekko，然后安装AutoGPTQ

```
Running setup.py develop for auto-gptq
    error: subprocess-exited-with-error
    
    × python setup.py develop did not run successfully.
    │ exit code: 1
    ╰─> [75 lines of output]
        Generating qigen kernels...
        conda_cuda_include_dir /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/nvidia/cuda_runtime/include
        running develop
        running egg_info
        writing auto_gptq.egg-info/PKG-INFO
        writing dependency_links to auto_gptq.egg-info/dependency_links.txt
        writing requirements to auto_gptq.egg-info/requires.txt
        writing top-level names to auto_gptq.egg-info/top_level.txt
        /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/utils/cpp_extension.py:502: UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.
          warnings.warn(msg.format('we could not find ninja.'))
        reading manifest file 'auto_gptq.egg-info/SOURCES.txt'
        adding license file 'LICENSE'
        writing manifest file 'auto_gptq.egg-info/SOURCES.txt'
        running build_ext
        /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/utils/cpp_extension.py:414: UserWarning: The detected CUDA version (11.0) has a minor version mismatch with the version that was used to compile PyTorch (11.8). Most likely this shouldn't be a problem.
          warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
        building 'autogptq_cuda_64' extension
        g++ -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/tangnn/anaconda3/envs/llmmodel/include -fPIC -O2 -isystem /home/tangnn/anaconda3/envs/llmmodel/include -fPIC -I/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include -I/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/TH -I/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -Iautogptq_cuda -I/home/tangnn/anaconda3/envs/llmmodel/include/python3.10 -I/home/tangnn/anaconda3/envs/llmmodel/include/python3.10 -c autogptq_extension/cuda_64/autogptq_cuda_64.cpp -o build/temp.linux-x86_64-3.10/autogptq_extension/cuda_64/autogptq_cuda_64.o -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011" -DTORCH_EXTENSION_NAME=autogptq_cuda_64 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17
        In file included from /home/tangnn/anaconda3/envs/llmmodel/include/python3.10/pyport.h:210:0,
                         from /home/tangnn/anaconda3/envs/llmmodel/include/python3.10/Python.h:50,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/python_headers.h:12,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/Device.h:4,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/python.h:8,
                         from autogptq_extension/cuda_64/autogptq_cuda_64.cpp:2:
        /usr/local/include/c++/7.1.0/math.h:65:12: error: ‘constexpr bool std::isinf(double)’ conflicts with a previous declaration
         using std::isinf;
                    ^~~~~
        In file included from /usr/include/features.h:375:0,
                         from /usr/local/include/c++/7.1.0/x86_64-pc-linux-gnu/bits/os_defines.h:39,
                         from /usr/local/include/c++/7.1.0/x86_64-pc-linux-gnu/bits/c++config.h:533,
                         from /usr/local/include/c++/7.1.0/functional:48,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/c10/core/DeviceType.h:10,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/c10/core/Device.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:11,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/ATen/core/Tensor.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/ATen/Tensor.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/function_hook.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/cpp_hook.h:2,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/variable.h:6,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/autograd.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/autograd.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/all.h:7,
                         from autogptq_extension/cuda_64/autogptq_cuda_64.cpp:1:
        /usr/include/bits/mathcalls.h:202:1: note: previous declaration ‘int isinf(double)’
         __MATHDECL_1 (int,isinf,, (_Mdouble_ __value)) __attribute__ ((__const__));
         ^
        In file included from /home/tangnn/anaconda3/envs/llmmodel/include/python3.10/pyport.h:210:0,
                         from /home/tangnn/anaconda3/envs/llmmodel/include/python3.10/Python.h:50,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/python_headers.h:12,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/Device.h:4,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/python.h:8,
                         from autogptq_extension/cuda_64/autogptq_cuda_64.cpp:2:
        /usr/local/include/c++/7.1.0/math.h:66:12: error: ‘constexpr bool std::isnan(double)’ conflicts with a previous declaration
         using std::isnan;
                    ^~~~~
        In file included from /usr/include/features.h:375:0,
                         from /usr/local/include/c++/7.1.0/x86_64-pc-linux-gnu/bits/os_defines.h:39,
                         from /usr/local/include/c++/7.1.0/x86_64-pc-linux-gnu/bits/c++config.h:533,
                         from /usr/local/include/c++/7.1.0/functional:48,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/c10/core/DeviceType.h:10,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/c10/core/Device.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/ATen/core/TensorBody.h:11,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/ATen/core/Tensor.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/ATen/Tensor.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/function_hook.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/cpp_hook.h:2,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/variable.h:6,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/autograd/autograd.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/autograd.h:3,
                         from /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/all.h:7,
                         from autogptq_extension/cuda_64/autogptq_cuda_64.cpp:1:
        /usr/include/bits/mathcalls.h:235:1: note: previous declaration ‘int isnan(double)’
         __MATHDECL_1 (int,isnan,, (_Mdouble_ __value)) __attribute__ ((__const__));
         ^
        error: command '/usr/local/bin/g++' failed with exit code 1
        [end of output]
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
```

最后的解决方案：找到AtuoGPTQ的GitHub地址，然后再github上寻找是否有编译好的安装包

![image-20240115102805271](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115102805271.png)

从huggingface.github.iourl上进行安装，但安装的是cuda11.8，我们cuda是11.7的，查看huggingface.github.io地址：

![image-20240115103005085](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115103005085.png)

查看auto-gptq:

![image-20240115103041912](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115103041912.png)

直接修改url为cull7：

![image-20240115103130960](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115103130960.png)

让根据python版本进行下载，上传到linux上，然后进行whl文件安装：

```
python -m pip install auto_gptq-0.4.2+cu117-cp310-cp310-linux_x86_64.whl
```

然后安装peft

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ git clone https://github.com/huggingface/peft.git
Cloning into 'peft'...
fatal: unable to access 'https://github.com/huggingface/peft.git/': Failed connect to github.com:443; Connection timed out
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ git clone https://github.com/huggingface/peft.git
Cloning into 'peft'...
fatal: unable to access 'https://github.com/huggingface/peft.git/': Encountered end of file
```

先去github上查看是否有编译下的文件，发现并没有。然后去pypi.org上搜索peft，发现可直接pip 安装

![image-20240115103627260](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115103627260.png)

直接在Linux虚拟环境中安装peft

安装requirements.txt文件上的包：pip install -r requirements.txt



下面开始Gradio快速搭建问答平台：

基于gradio搭建的问答界面，实现了流式的输出，将下面代码复制到控制台运行，以下代码以Atom-7B模型为例，不同模型只需修改一下代码里的模型名称就好了

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ python examples/chat_gradio.py --model_name_or_path FlagAlpha/Atom-7B
Traceback (most recent call last):
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connection.py", line 203, in _new_conn
    sock = connection.create_connection(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/util/connection.py", line 85, in create_connection
    raise err
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/util/connection.py", line 73, in create_connection
    sock.connect(sa)
TimeoutError: timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connectionpool.py", line 790, in urlopen
    response = self._make_request(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connectionpool.py", line 491, in _make_request
    raise new_e
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connectionpool.py", line 467, in _make_request
    self._validate_conn(conn)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connectionpool.py", line 1096, in _validate_conn
    conn.connect()
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connection.py", line 611, in connect
    self.sock = sock = self._new_conn()
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/urllib3/connection.py", line 212, in _new_conn
    raise ConnectTimeoutError(
urllib3.exceptions.ConnectTimeoutError: (<urllib3.connection.HTTPSConnection object at 0x7f84de1481f0>, 'Connection to huggingface.co timed out. (connect timeout=10)')
```

因传入的参数为--model_name_or_path FlagAlpha/Atom-7B，连接失败。

然后再github上找到一个Atom-7B的模型：

![image-20240115105658457](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115105658457.png)

使用上诉语句进行安装，然而并没有成功。

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ python examples/chat_gradio.py --model_name_or_path https://github.com/HuengchI/Atom-7B
Traceback (most recent call last):
  File "/data/tangnn/other_projects/Llama2-Chinese-main/examples/chat_gradio.py", line 83, in <module>
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=False)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 652, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 496, in get_tokenizer_config
    resolved_config_file = cached_file(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/hub.py", line 417, in cached_file
    resolved_file = hf_hub_download(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 110, in _inner_fn
    validate_repo_id(arg_value)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 158, in validate_repo_id
    raise HFValidationError(
huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'https://github.com/HuengchI/Atom-7B'. Use `repo_type` argument if needed.
```



> gradio官方文档  [Gradio Interface Docs](https://www.gradio.app/docs/interface)



下载huggingface下Atom-7B的所有文件[FlagAlpha/Atom-7B at main (huggingface.co)](https://huggingface.co/FlagAlpha/Atom-7B/tree/main)

并把文件放入到linux中以下目录中，接着运行以下命令，出现如下结果：

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ python examples/chat_gradio.py --model_name_or_path /data/tangnn/other_projects/Atom-7B
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda114.so
False
/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /home/tangnn/anaconda3/envs/llmmodel did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/home/tangnn/cplex/CPLEX_Studio128cplex/bin/x86-64_linux')}
  warn(msg)
/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /home/tangnn/cplex/CPLEX_Studio128cplex/bin/x86-64_linux:/home/tangnn/cplex/CPLEX_Studio128/cpoptimizer/bin/x86-64_linux: did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/usr/local/cuda/lib64/libcudart.so'), PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0')}.. We'll flip a coin and try one of these, in order to fail forward.
Either way, this might cause trouble in the future:
If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 114
CUDA SETUP: Required library version not found: libbitsandbytes_cuda114.so. Maybe you need to compile it from source?
CUDA SETUP: Defaulting to libbitsandbytes_cpu.so...

================================================ERROR=====================================
CUDA SETUP: CUDA detection failed! Possible reasons:
1. CUDA driver not installed
2. CUDA not installed
3. You have multiple conflicting CUDA libraries
4. Required library not pre-compiled for this bitsandbytes release!
CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.
CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.
================================================================================

CUDA SETUP: Something unexpected happened. Please compile from source:
git clone git@github.com:TimDettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=114 make cuda11x
python setup.py install
CUDA SETUP: Setup Failed!
Traceback (most recent call last):
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1099, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 32, in <module>
    from ...modeling_utils import PreTrainedModel
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/modeling_utils.py", line 86, in <module>
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/__init__.py", line 3, in <module>
    from .accelerator import Accelerator
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/accelerator.py", line 35, in <module>
    from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/checkpointing.py", line 24, in <module>
    from .utils import (
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/utils/__init__.py", line 131, in <module>
    from .bnb import has_4bit_bnb_layers, load_and_quantize_model
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/utils/bnb.py", line 42, in <module>
    import bitsandbytes as bnb
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/__init__.py", line 6, in <module>
    from . import cuda_setup, utils, research
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/research/__init__.py", line 1, in <module>
    from . import nn
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/research/nn/__init__.py", line 1, in <module>
    from .modules import LinearFP8Mixed, LinearFP8Global
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/research/nn/modules.py", line 8, in <module>
    from bitsandbytes.optim import GlobalOptimManager
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/optim/__init__.py", line 6, in <module>
    from bitsandbytes.cextension import COMPILED_WITH_CUDA
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cextension.py", line 20, in <module>
    raise RuntimeError('''
RuntimeError: 
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/tangnn/other_projects/Llama2-Chinese-main/examples/chat_gradio.py", line 86, in <module>
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 492, in from_pretrained
    model_class = _get_model_class(config, cls._model_mapping)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 376, in _get_model_class
    supported_models = model_mapping[type(config)]
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 666, in __getitem__
    return self._load_attr_from_module(model_type, model_name)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 680, in _load_attr_from_module
    return getattribute_from_module(self._modules[module_name], attr)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 625, in getattribute_from_module
    if hasattr(module, attr):
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1089, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1101, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):

        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```

查看路径LD_LIBRARY_PATH:

![image-20240115171618363](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115171618363.png)



```
(llmmodel) [tangnn@cqi145-daas-gpu03 other_projects]$ echo $LD_LIBRARY_PATH
/home/tangnn/cplex/CPLEX_Studio128cplex/bin/x86-64_linux:/home/tangnn/cplex/CPLEX_Studio128/cpoptimizer/bin/x86-64_linux:
```

里面并没有cuda，通过用户目录下，vi .bashrc 进行添加

![image-20240115171933587](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115171933587.png)

接着运行python examples/chat_gradio.py --model_name_or_path /data/tangnn/other_projects/Atom-7B，还是没有成功，考虑到可能是cuda版本问题，因为我们安装的是cuda11.8，因此现卸载torch、torchvision以及torchaudio

```
pip install torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl
pip install torchaudio-2.0.2+cu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.2+cu117-cp310-cp310-linux_x86_64.whl
```

更新后torch版本后，进行查看torch是否导入成功

```
(llmmodel) [tangnn@cqi145-daas-gpu03 downloads]$ python 
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/__init__.py", line 229, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so: symbol cublasSetWorkspace_v2, version libcublas.so.11 not defined in file libcublas.so.11 with link time reference
>>> exit()
```

发现并未导入成功，然后路径PATH增加export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}

![image-20240115173203239](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115173203239.png)

接着刷新环境即可

```
(llmmodel) [tangnn@cqi145-daas-gpu03 ~]$ vi .bashrc
(llmmodel) [tangnn@cqi145-daas-gpu03 ~]$ source .bashrc
(base) [tangnn@cqi145-daas-gpu03 ~]$ conda activate llmmodel
(llmmodel) [tangnn@cqi145-daas-gpu03 ~]$ python
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> exit()
```

接着再次进行llama2目录下，运行命令，仍报如下错误：

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ python examples/chat_gradio.py --model_name_or_path /data/tangnn/other_projects/Atom-7B
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda114.so
False
/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /home/tangnn/anaconda3/envs/llmmodel did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/home/tangnn/cplex/CPLEX_Studio128cplex/bin/x86-64_linux')}
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda-11.4/lib64/libcudart.so.11.0
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 114
CUDA SETUP: Required library version not found: libbitsandbytes_cuda114.so. Maybe you need to compile it from source?
CUDA SETUP: Defaulting to libbitsandbytes_cpu.so...

================================================ERROR=====================================
CUDA SETUP: CUDA detection failed! Possible reasons:
1. CUDA driver not installed
2. CUDA not installed
3. You have multiple conflicting CUDA libraries
4. Required library not pre-compiled for this bitsandbytes release!
CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.
CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.
================================================================================

CUDA SETUP: Something unexpected happened. Please compile from source:
git clone git@github.com:TimDettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=114 make cuda11x
python setup.py install
CUDA SETUP: Setup Failed!
Traceback (most recent call last):
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1099, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 32, in <module>
    from ...modeling_utils import PreTrainedModel
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/modeling_utils.py", line 86, in <module>
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/__init__.py", line 3, in <module>
    from .accelerator import Accelerator
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/accelerator.py", line 35, in <module>
    from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/checkpointing.py", line 24, in <module>
    from .utils import (
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/utils/__init__.py", line 131, in <module>
    from .bnb import has_4bit_bnb_layers, load_and_quantize_model
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/utils/bnb.py", line 42, in <module>
    import bitsandbytes as bnb
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/__init__.py", line 6, in <module>
    from . import cuda_setup, utils, research
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/research/__init__.py", line 1, in <module>
    from . import nn
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/research/nn/__init__.py", line 1, in <module>
    from .modules import LinearFP8Mixed, LinearFP8Global
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/research/nn/modules.py", line 8, in <module>
    from bitsandbytes.optim import GlobalOptimManager
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/optim/__init__.py", line 6, in <module>
    from bitsandbytes.cextension import COMPILED_WITH_CUDA
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/bitsandbytes/cextension.py", line 20, in <module>
    raise RuntimeError('''
RuntimeError: 
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/tangnn/other_projects/Llama2-Chinese-main/examples/chat_gradio.py", line 86, in <module>
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 492, in from_pretrained
    model_class = _get_model_class(config, cls._model_mapping)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 376, in _get_model_class
    supported_models = model_mapping[type(config)]
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 666, in __getitem__
    return self._load_attr_from_module(model_type, model_name)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 680, in _load_attr_from_module
    return getattribute_from_module(self._modules[module_name], attr)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 625, in getattribute_from_module
    if hasattr(module, attr):
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1089, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1101, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):

        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```

发现可能是bitsandbytes的问题，按网络教程安装bitsandbytes，报错nvcc fatal   : Unsupported gpu architecture 'compute_86'

```
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ cd bitsandbytes/
(llmmodel) [tangnn@cqi145-daas-gpu03 bitsandbytes] (main)$ CUDA_VERSION=114 make cuda11x
ENVIRONMENT
============================
CUDA_VERSION: 114
============================
NVCC path: /usr/local/cuda-11.0/bin/nvcc
GPP path: /usr/bin/g++ VERSION: g++ (GCC) 7.1.0
CUDA_HOME: /usr/local/cuda-11.0
CONDA_PREFIX: /home/tangnn/anaconda3/envs/llmmodel
PATH: /usr/local/cuda-11.0/bin:/home/tangnn/anaconda3/envs/llmmodel/bin:/home/tangnn/anaconda3/condabin:/usr/local/cuda-11.0/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/tangnn/cplex/cplex/bin/x86-64_linux:/home/tangnn/cplex/cpoptimizer/bin/x86-64_linux:/home/tangnn/.local/bin:/home/tangnn/bin:/home/tangnn/cplex/cplex/bin/x86-64_linux:/home/tangnn/cplex/cpoptimizer/bin/x86-64_linux:/home/tangnn/cplex/cplex/bin/x86-64_linux:/home/tangnn/cplex/cpoptimizer/bin/x86-64_linux:/usr/local/cuda-11.4/bin
LD_LIBRARY_PATH: /usr/local/cuda-11.4/lib64:/home/tangnn/cplex/CPLEX_Studio128cplex/bin/x86-64_linux:/home/tangnn/cplex/CPLEX_Studio128/cpoptimizer/bin/x86-64_linux:
============================
/usr/local/cuda-11.0/bin/nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc /data/tangnn/other_projects/Llama2-Chinese-main/bitsandbytes/csrc/ops.cu /data/tangnn/other_projects/Llama2-Chinese-main/bitsandbytes/csrc/kernels.cu -I /usr/local/cuda-11.0/include -I /data/tangnn/other_projects/Llama2-Chinese-main/bitsandbytes/csrc -I /home/tangnn/anaconda3/envs/llmmodel/include -I /data/tangnn/other_projects/Llama2-Chinese-main/bitsandbytes/include -L /usr/local/cuda-11.0/lib64 -lcudart -lcublas -lcublasLt -lcusparse -L /home/tangnn/anaconda3/envs/llmmodel/lib --output-directory /data/tangnn/other_projects/Llama2-Chinese-main/bitsandbytes/build
nvcc fatal   : Unsupported gpu architecture 'compute_86'
make: *** [cuda11x] Error 1
```

接着在bitsandbytes  github的issue中搜素compute_86:

[Issues · TimDettmers/bitsandbytes (github.com)](https://github.com/TimDettmers/bitsandbytes/issues?q=compute_86)

![image-20240115173831889](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115173831889.png)

有解决方案说是直接更新bitsandbytes即可。

![image-20240115174050816](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240115174050816.png)

接着对其进行更新：

```
(llmmodel) [tangnn@cqi145-daas-gpu03 bitsandbytes] (main)$ pip install -U bitsandbytes
(llmmodel) [tangnn@cqi145-daas-gpu03 Llama2-Chinese-main]$ python examples/chat_gradio.py --model_name_or_path /data/tangnn/other_projects/Atom-7B
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.58s/it]
Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```





运行pretrain_clm.py文件：

```
(llmmodel) [tangnn@cqi145-daas-gpu03 pretrain]$ python pretrain_clm.py --output_dir ./output_model  --model_name_or_path /data/tangnn/other_projects/Atom-7B  --train_files ../../data/train_sft.csv   --validation_files  ../../data/dev_sft.csv ../../data/dev_sft_sharegpt.csv --do_train --overwrite_output_dir

Traceback (most recent call last):
  File "/data/tangnn/other_projects/Llama2-Chinese-main/train/pretrain/pretrain_clm.py", line 613, in <module>
    main()
  File "/data/tangnn/other_projects/Llama2-Chinese-main/train/pretrain/pretrain_clm.py", line 553, in main
    trainer = Trainer(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 499, in __init__
    self._move_model_to_device(model, args.device)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 741, in _move_model_to_device
    model = model.to(device)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/modeling_utils.py", line 1900, in to
    return super().to(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1145, in to
    return self._apply(convert)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1143, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1016.00 MiB (GPU 0; 39.59 GiB total capacity; 25.24 GiB already allocated; 379.19 MiB free; 25.26 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```



```
(llmmodel) [tangnn@cqi145-daas-gpu03 pretrain]$ CUDA_VISIBLE_DEVICES=1 python pretrain_clm.py --output_dir ./output_model  --model_name_or_path /data/tangnn/other_projects/Atom-7B  --train_files ../../data/train_sft.csv   --validation_files  ../../data/dev_sft.csv ../../data/dev_sft_sharegpt.csv --do_train --overwrite_output_dir

Traceback (most recent call last):
  File "/data/tangnn/other_projects/Llama2-Chinese-main/train/pretrain/pretrain_clm.py", line 613, in <module>
    main()
  File "/data/tangnn/other_projects/Llama2-Chinese-main/train/pretrain/pretrain_clm.py", line 574, in main
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 1539, in train
    return inner_training_loop(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 1809, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 2654, in training_step
    loss = self.compute_loss(model, inputs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 2679, in compute_loss
    outputs = model(**inputs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 806, in forward
    outputs = self.model(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 693, in forward
    layer_outputs = decoder_layer(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 421, in forward
    hidden_states = self.mlp(hidden_states)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 216, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 344.00 MiB (GPU 0; 39.59 GiB total capacity; 37.89 GiB already allocated; 62.19 MiB free; 38.26 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```



```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1016.00 MiB (GPU 1; 39.59 GiB total capacity; 25.12 GiB already allocated; 451.19 MiB free; 25.18 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

```

![image-20240117112101901](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240117112101901.png)

```
(llmmodel) [tangnn@cqi145-daas-gpu03 pretrain]$ CUDA_VISIBLE_DEVICES=1,0 python pretrain_clm.py --output_dir ./output_model  --model_name_or_path /data/tangnn/other_projects/Atom-7B-chat  --train_files ../../data/train_sft.csv   --validation_files  ../../data/dev_sft.csv ../../data/dev_sft_sharegpt.csv --per_device_train_batch_size 1  --do_train --overwrite_output_dir
```

使用deepspeed出现的错误：

![image-20240117153447985](C:\Users\Nana.Tang1\AppData\Roaming\Typora\typora-user-images\image-20240117153447985.png)

```
ValueError: Expected a string path to an existing deepspeed config, or a dictionary, or a base64 encoded string. Received: ../ds_config_zero3.json
```







gcc版本不对，尝试源码安装gcc11.4

```
$ mkdir tmp && cd tmp
$ sudo ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib  --enable-bootstrap  --prefix=/usr/local/gcc-11
checking build system type... x86_64-pc-linux-gnu
checking host system type... x86_64-pc-linux-gnu
checking target system type... x86_64-pc-linux-gnu
checking for a BSD-compatible install... /bin/install -c
checking whether ln works... yes
checking whether ln -s works... yes
checking for a sed that does not truncate output... /bin/sed
checking for gawk... gawk
checking for libatomic support... yes
checking for libitm support... yes
checking for libsanitizer support... yes
checking for libvtv support... yes
checking for libhsail-rt support... yes
checking for libphobos support... yes
checking for gcc... gcc
checking whether the C compiler works... yes
checking for C compiler default output file name... a.out
checking for suffix of executables... 
checking whether we are cross compiling... no
checking for suffix of object files... o
checking whether we are using the GNU C compiler... yes
checking whether gcc accepts -g... yes
checking for gcc option to accept ISO C89... none needed
checking for g++... g++
checking whether we are using the GNU C++ compiler... yes
checking whether g++ accepts -g... yes
checking whether g++ accepts -static-libstdc++ -static-libgcc... yes
checking for gnatbind... no
checking for gnatmake... no
checking whether compiler driver understands Ada... no
checking how to compare bootstrapped objects... cmp --ignore-initial=16 $$f1 $$f2
checking for objdir... .libs
checking for the correct version of gmp.h... no
configure: error: Building GCC requires GMP 4.2+, MPFR 3.1.0+ and MPC 0.8.0+.
Try the --with-gmp, --with-mpfr and/or --with-mpc options to specify
their locations.  Source code for these libraries can be found at
their respective hosting sites as well as at
https://gcc.gnu.org/pub/gcc/infrastructure/.  See also
http://gcc.gnu.org/install/prerequisites.html for additional info.  If
you obtained GMP, MPFR and/or MPC from a vendor distribution package,
make sure that you have installed both the libraries and the header

```

报错

```
configure: error: Building GCC requires GMP 4.2+, MPFR 3.1.0+ and MPC 0.8.0+.
Try the --with-gmp, --with-mpfr and/or --with-mpc options to specify
their locations.  Source code for these libraries can be found at
their respective hosting sites as well as at
https://gcc.gnu.org/pub/gcc/infrastructure/.  See also
http://gcc.gnu.org/install/prerequisites.html for additional info.  If
you obtained GMP, MPFR and/or MPC from a vendor distribution package,
make sure that you have installed both the libraries and the header
```

解决办法如下，手动安装依赖

```
tar -xvf gmp-6.1.0.tar.bz2
cd gmp-6.1.0/
./configure --prefix=/usr/local/gmp-6.1.0
make -j $(nproc)
sudo make install

tar -xvf mpfr-3.1.6.tar.bz2
cd mpfr-3.1.6/
./configure --prefix=/usr/local/mpfr-3.1.6 --with-gmp=/usr/local/gmp-6.1.0
make -j $(nproc)
sudo make install

tar -xvf mpc-1.0.3.tar.gz
cd mpc-1.0.3/
./configure --prefix=/usr/local/mpc-1.0.3 --with-gmp=/usr/local/gmp-6.1.0 --with-mpfr=/usr/local/mpfr-3.1.6
make -j $(nproc)
sudo make install
```





再次安装：

```bash
sudo ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib  --enable-bootstrap  --prefix=/usr/local/gcc-11 --with-gmp=/usr/local/gmp-6.1.0 --with-mpfr=/usr/local/mpfr-3.1.6 --with-mpc=/usr/local/mpc-1.0.3
```

执行成功后在`tmp`目录下会创建一个`Makefile`文件.

安装：

```
make -j $(nproc)
sudo make install
```









安装g++编译报错，需要升级gdb

```
wget http://ftp.gnu.org/gnu/gdb/gdb-10.2.tar.xz
tar -Jxvf gdb-10.2.tar.xz

cd gdb-10.2
 
./configure --prefix=/usr/local/gdb-10.2
 
make -j $(nproc)
 
sudo make install
```

编译gdb报错

```
WARNING: 'makeinfo' is missing on your system.
         You should only need it if you modified a '.texi' file, or
         any other file indirectly affecting the aspect of the manual.
         You might want to install the Texinfo package:
         <http://www.gnu.org/software/texinfo/>
         The spurious makeinfo call might also be the consequence of
         using a buggy 'make' (AIX, DU, IRIX), in which case you might
         want to install GNU make:
         <http://www.gnu.org/software/make/>
make[4]: *** [annotate.info] Error 127
make[4]: Leaving directory `/data/tangnn/other_projects/gdb-10.2/gdb/doc'
make[3]: *** [subdir_do] Error 1
make[3]: Leaving directory `/data/tangnn/other_projects/gdb-10.2/gdb'
make[2]: *** [all] Error 2
make[2]: Leaving directory `/data/tangnn/other_projects/gdb-10.2/gdb'
make[1]: *** [all-gdb] Error 2
make[1]: Leaving directory `/data/tangnn/other_projects/gdb-10.2'
make: *** [all] Error 2
```

报没有`makeinfo`，安装texinfo

```
yum install texinfo
```

接着编译gdb



编译gdb后安装gdb，完成后，再接着编译gcc，报错如下：

```
g++: error: unrecognized command line option ‘-no-pie’
make[3]: *** [gcov] Error 1
make[3]: *** Waiting for unfinished jobs....
g++: error: unrecognized command line option ‘-no-pie’
make[3]: *** [gcov-dump] Error 1
make[3]: Leaving directory `/data/tangnn/other_projects/gcc-11.4.0/tmp/gcc'
make[2]: *** [all-stage1-gcc] Error 2
make[2]: Leaving directory `/data/tangnn/other_projects/gcc-11.4.0/tmp'
make[1]: *** [stage1-bubble] Error 2
make[1]: Leaving directory `/data/tangnn/other_projects/gcc-11.4.0/tmp'
make: *** [all] Error 2
```



```
Traceback (most recent call last):
  File "/data/tangnn/other_projects/Llama2-Chinese-main/train/pretrain/pretrain_clm.py", line 613, in <module>
    main()
  File "/data/tangnn/other_projects/Llama2-Chinese-main/train/pretrain/pretrain_clm.py", line 574, in main
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 1539, in train
    return inner_training_loop(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/transformers/trainer.py", line 1659, in _inner_training_loop
    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/accelerator.py", line 1198, in prepare
    result = self._prepare_deepspeed(*args)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/accelerate/accelerator.py", line 1537, in _prepare_deepspeed
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/__init__.py", line 171, in initialize
    engine = DeepSpeedEngine(args=args,
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 304, in __init__
    self._configure_optimizer(optimizer, model_parameters)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 1234, in _configure_optimizer
    self.optimizer = self._configure_zero_optimizer(basic_optimizer)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 1563, in _configure_zero_optimizer
    optimizer = DeepSpeedZeroOptimizer_Stage3(
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py", line 362, in __init__
    self._setup_for_real_optimizer()
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py", line 474, in _setup_for_real_optimizer
    self.initialize_optimizer_states()
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py", line 1008, in initialize_optimizer_states
    self._optimizer_step(i)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/runtime/zero/stage3.py", line 934, in _optimizer_step
    self.optimizer.step()
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/torch/optim/optimizer.py", line 280, in wrapper
    out = func(*args, **kwargs)
  File "/home/tangnn/anaconda3/envs/llmmodel/lib/python3.10/site-packages/deepspeed/ops/adam/fused_adam.py", line 155, in step
    state['exp_avg'] = torch.zeros_like(p.data)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.80 GiB (GPU 0; 39.59 GiB total capacity; 23.45 GiB already allocated; 2.16 GiB free; 23.46 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

```







set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

![image-20240119113358602](python基础知识点.assets/image-20240119113358602.png)





##### 问题ds_opt_adam

```
RuntimeError: Ninja is required to load C++ extensions
AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
```

未发现这个属性：ds_opt_adam，对deepspeed进行更新：`pip install deepspeed --upgrade`，更新后cpu_adam与fused_adam并没有编译，这是运行deep speed时，如果gcc版本在7.2-11区间内，则便运行便编译，否则会报错，无cpu_adam、fused_adam。

另解决方法，在训练文件头部加：

```
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
```

deepspeed 编译cpu_adam，fused_adam，需要将gcc切换到版本8-11以内

切换gcc到版本9：`scl enable devtoolset-11 bash`，然后对deepspeed进行更新。

`DS_BUILD_FUSED_ADAM=1  DS_BUILD_CPU_ADAM=1 pip install deepspeed --global-option="build_ext" --global-option="-j96"`





·'2.1.2+cu121'

 `CUDA Version: 11.4`

   --max_split_size_mb 32
    --tensor-parallel-size 4

```
curl http://localhost:8099/v1/models
```



```
curl http://localhost:8099/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```



CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server  --model /data/litao/workspace/models/Atom-7B-Chat --port 8099

```
CUDA_VISIBLE_DEVICES=1  python -m vllm.entrypoints.openai.api_server  --model /data/litao/workspace/models/Qwen-7B-Chat --port 8099 --trust-remote-code

CUDA_VISIBLE_DEVICES=1  python -m vllm.entrypoints.openai.api_server  --model /data/tangnn/downloads/models/chatglm-6b-int4 --port 8099 --trust-remote-code

CUDA_VISIBLE_DEVICES=1  python -m vllm.entrypoints.api_server  --model /data/litao/workspace/models/Qwen-7B-Chat --port 8099 --trust-remote-code

CUDA_VISIBLE_DEVICES=1  python -m vllm.entrypoints.api_server  --model /data/litao/workspace/models/Atom-7B-Chat --port 8099 --trust-remote-code
```





```
CUDA_VISIBLE_DEVICES=1  python -m fastchat.serve.cli --model-path /data/tangnn/downloads/models/bge-base-zh  

CUDA_VISIBLE_DEVICES=1  python -m fastchat.serve.model_worker --model-path /data/tangnn/downloads/models/bge-base-zh --controller http://localhost:8099 --port 8099 --worker http://localhost:8088






python -m fastchat.serve.controller --port 21000 > logs/fastchat.log 2>&1 &

python -m fastchat.serve.model_worker --model-path /data/litao/workspace/models/bge-large-zh-v1.5  > logs/beg_fschat.log 2>&1 &

python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 3344  > logs/fastchat.log 2>&1 &
```





```
python -m fastchat.serve.controller  --host "0.0.0.0" > logs/fastchat.log 2>&1 &

python -m fastchat.serve.model_worker  --model-path /data/tangnn/downloads/models/bge-base-zh --host "0.0.0.0"  > logs/beg_fschat.log 2>&1 &
python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8077 > logs/fastchat.log 2>&1 &
```





```
CUDA_VISIBLE_DEVICES=1  python -m vllm.entrypoints.openai.api_server  --model /data/tangnn/downloads/models/Qwen-7B-Chat-Int4 --port 8099 --trust-remote-code
```



![image-20240314021538175](python基础知识点.assets/image-20240314021538175.png)

```
python -m fastchat.serve.controller  --host "0.0.0.0" --port 21004 > logs/fastchat.log 2>&1 &

python -m fastchat.serve.multi_model_worker  --model-path /data/tangnn/downloads/models/bge-base-zh --model-name bge-base-zh  --model-path /data/litao/workspace/models/Qwen-7B-Chat --model-name Qwen-7B-Chat --model-path /data/litao/workspace/models/m3e-base --model-name m3e-base  --host "0.0.0.0" --port 21005 --controller-address "http://0.0.0.0:21004"  > logs/beg_fschat.log 2>&1 &

python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8066 --controller-address "http://0.0.0.0:21004"  > logs/fastchat.log 2>&1 &


python -m fastchat.serve.controller  --host "0.0.0.0" > logs/fastchat_serve.log 2>&1 &

python -m fastchat.serve.multi_model_worker  --model-path /data/tangnn/downloads/models/bge-base-zh --model-name bge-base-zh --model-path /data/litao/workspace/models/m3e-base --model-name m3e-base  --model-path /data/litao/workspace/models/Qwen-7B-Chat --model-name Qwen-7B-Chat   --host "0.0.0.0" > logs/two_model_fschat.log 2>&1 &

python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8077 > logs/fastchat_openai.log 2>&1 &





python -m fastchat.serve.controller  --host "0.0.0.0" --port 21006 > logs/one_model/fastchat_controller.log 2>&1 &

python -m fastchat.serve.model_worker  --model-path /data/litao/workspace/models/m3e-base --model-name m3e-base --host "0.0.0.0" --port 21007 --controller-address "http://0.0.0.0:21006" > logs/one_model/m3e_fschat.log 2>&1 &
python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8079 --controller-address "http://0.0.0.0:21006" > logs/one_model/fastchat.log 2>&1 &


```

