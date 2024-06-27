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



![image-20220505092823394](C:\Users\XX\AppData\Roaming\Typora\typora-user-images\image-20220505092823394.png)



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

![image-20220507145905413](C:\UsersXX\AppData\Roaming\Typora\typora-user-images\image-20220507145905413.png)

查看conda虚拟环境

```
./conda env list
```

![image-20220507150100344](C:\Users\XX\AppData\Roaming\Typora\typora-user-images\image-20220507150100344.png)

创建conda虚拟环境

```
./conda create -n welding-parameter python=3.7.11
```

![image-20220507150631626](C:\Users\XX\AppData\Roaming\Typora\typora-user-images\image-20220507150631626.png)



激活虚拟环境

```
source activate welding-parameter
```

![image-20220507154018365](C:\UsersXX\AppData\Roaming\Typora\typora-user-images\image-20220507154018365.png)



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
[ bin]$ conda config --add channels http://mirrors.aliyun.com/anaconda/pkgs/r
[ bin]$ conda config --add channels http://mirrors.aliyun.com/anaconda/pkgs/msys2
[ bin]$ conda config --set show_channel_urls yes
[ bin]$ conda config --show
add_anaconda_token: True
add_pip_as_python_dependency: True


```
pip install gunicorn -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gunicorn
```

```
[gpu03 bin]$ pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
Writing to /home/XX/.config/pip/pip.conf
[gpu03 bin]$ pip config set install.trusted-host mirrors.aliyun.com
Writing to /home/XX/.config/pip/pip.conf
```

设置pip，给pip添加镜像源，然后用pip install  包 进行安装



##### 问题1

```
Can't get remote credentials for deployment server XX@00.00.00.00:22 password
```

删除多余的解释器，仅留一个



##### 问题2

![image-20220510105654967](C:\Users\XX\AppData\Roaming\Typora\typora-user-images\image-20220510105654967.png)

python -m pip install dask distributed --upgrade





