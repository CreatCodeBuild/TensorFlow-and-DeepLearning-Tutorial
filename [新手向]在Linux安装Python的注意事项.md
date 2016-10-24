## 在Linux安装Python之注意事项

### 1.Ubuntu装机自带Python2与Python3
/usr/bin 路径下有自带的Python

### 2.最有用的 3 个 Shell 指令
```bash
type python
type -a python

which python

whereis python
```
三个指令的功能有一些不同，但是都很有用

### 3.什么是环境变量 Environmental Variable
环境变量是指的，在命令行输入任何指令时，Shell在电脑的哪些路径中去寻找你所敲的指令。除了 shell builtin 以外，每一个指令都是一个具体的程序（也就是说你可以在系统的各种路径中找到）

每个用户都可以在 /home/你的用户名 下的 .bashrc 文件中找到你的环境变量。这个文件是隐藏文件。不要被后缀名吓到，只是一个txt而已。
```bash
# 你可以通过
ls -a
# 看到这个文件
gedit .bashrc  # 可以打开这个文件
```
你在 Files（Ubuntu自带的文件浏览器）里将“显示隐藏文件”选中也可以

### 4.安装Anaconda之注意事项

### 5.Virtual Environment 是什么？

[视频链接]()
