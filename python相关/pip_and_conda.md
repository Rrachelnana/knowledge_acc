## conda 与pip的区别与联系

联系：conda与pip都是包管理器，有一些重叠功能

区别：conda是一个跨平台的包和环境管理器，安装的包是二进制的，同时安装包不限于python包，还可以是c、c++库、R库以及其他软件；即python安装python包，conda可安装用任何语言编写的软件包；
使用pip之前必须先安装python解释器；而conda可直接安装python包和python解释器
conda能创建独立环境，可包含不同版本的python和包，而pip不支持内置独立环境，必须依赖其他工具，如virtual来创建隔离环境
conda和pip在实现环境依赖时也不同，pip不能确保所有包的依赖关系同时满足，而conda使用satisfiability (SAT) 来检验包的所有依赖是否满足


## 虚拟环境其他

anaconda3 清华源镜像：[Index of /anaconda/archive/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

查看conda虚拟环境
```
./conda env list
```


创建conda虚拟环境
```
conda create -n welding-parameter python=3.7.11
```

激活虚拟环境
```
source activate welding-parameter
```


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


实际命令操作
1.conda增加源
```
[ bin]$ conda config --add channels http://mirrors.aliyun.com/anaconda/pkgs/r
[ bin]$ conda config --add channels http://mirrors.aliyun.com/anaconda/pkgs/msys2
[ bin]$ conda config --set show_channel_urls yes
[ bin]$ conda config --show
add_anaconda_token: True
add_pip_as_python_dependency: True
```
2.通过豆瓣源使用pip安装gunicorn包
```
pip install gunicorn -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gunicorn
```
3.设置阿里源为主源
```
[gpu03 bin]$ pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
Writing to /home/XX/.config/pip/pip.conf
[gpu03 bin]$ pip config set install.trusted-host mirrors.aliyun.com
Writing to /home/XX/.config/pip/pip.conf
```


设置pip，给pip添加镜像源，然后用pip install  包 进行安装



