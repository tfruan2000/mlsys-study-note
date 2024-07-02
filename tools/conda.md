# conda



## 换源

vim ~/.condarc

```bash
show_channel_urls: true
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
auto_activate_base: false
```

## 使用

```bash
 conda create -n B --clone A       #克隆环境A来创建名为B的环境
 conda create -n B  python=3.10
 conda activate xxxx               #开启xxxx环境
 conda deactivate                  #关闭环境
 conda info -e                    #显示所有的虚拟环境
 conda remove -n xxxx --all       #删除已创建的xxxx虚拟环境
 
 conda update --all
 
 conda clean -p      #删除没有用的包
 conda clean -t      #tar打包
 conda clean -a
 
 conda config --show   #查看全部配置
```