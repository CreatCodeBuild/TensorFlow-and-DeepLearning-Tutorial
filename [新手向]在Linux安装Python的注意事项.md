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
所有运行着的程序都在某种“系统环境”中运行。Linux 的很多程序都是从 Shell 启动的。那么 Shell 就自带了一些环境变量。环境变量和编程语言里面的变量差不多，都是由 “名字” + “值” 组成的。比如，一个叫做 LOG_LEVEL 的变量可能等于 2；一个叫做 USER_NAME 的变量可能等于 “你的名字”。

环境变量有很多个，PATH 为其中一个。

PATH 指的是，在命令行输入任何指令时，Shell 在电脑的哪些路径中去寻找你所敲的指令。除了 shell builtin 以外，每一个指令都是一个具体的程序（也就是说你可以在系统的各种路径中找到）

每个用户都可以在 /home/你的用户名 下的 .bashrc 文件中找到你的环境变量。这个文件是隐藏文件。不要被后缀名吓到，只是一个txt而已。
```bash
# 你可以通过
ls -a
# 看到这个文件
gedit .bashrc  # 可以打开这个文件
```
你在 Files（Ubuntu自带的文件浏览器）里将“显示隐藏文件”选中也可以

将你要添加的路径粘贴到文件的最下方就可以了，如：
```
export PATH="/home/MyUserName/anaconda3/bin:$PATH"
```
注意，格式一定是 export PATH=“你要添加的路径:$PATH” !  
Bash 是一个简易的脚本语言，PATH 是一个变量，= 是赋值，：是String追加，$PATH 是引用 PATH 的值
```Bash
echo $PATH  # 你可以这样将 PATH 的值打印出来
```

### 4.安装Anaconda之注意事项

### 5.Virtual Environment 是什么？

[视频链接]()
