git克隆gitee上的项目报错443

1.windows中ssh地址：C:\Users\Nana.Tang1\.ssh
若当前没有id_rsa(私钥)、id_rsa.pub(公开钥)，需要生成

2.生成密钥：ssh-keygen

3.检查ssh是否能连通：ssh -T gitee.com

在gitee中添加公钥：设置--》SSH公钥--> 添加公钥（将id_rsa.pub中文本复制到公钥栏中，并给一个标题）

而后再进行克隆，报如下错：
kex_exchange_identification: Connection closed by remote host Connection closed by 198.18.0.252 port 22 Could not read from remote repository.  Please make sure you have the correct access rights and the repository exists.

