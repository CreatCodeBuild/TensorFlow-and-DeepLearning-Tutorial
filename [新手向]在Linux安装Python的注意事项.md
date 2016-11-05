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

### 3.基本 Python 与 Pip 的命令行使用
```
pip install package_name #  会下载一个包
```
但是，如果你有多个 python 版本，怎么确保下载对应的包呢？
```
python -m pip install package_name #  就会使用对应 python 的 pip 来下载
python3 -m pip install package_name
python2 -m pip install package_name
ThePythonYouWant -m pip install package_name
```
-m 代表 module。python -m module_name 意为使用 python 运行该模组。
pip 只是一个 python 模组，只是恰好有命令行快捷方式罢了。

### 4.什么是环境变量 Environmental Variable
所有运行着的程序都在某种“系统环境”中运行。Linux 的很多程序都是从 Shell 启动的。那么 Shell 就自带了一些环境变量。环境变量和编程语言里面的变量差不多，都是由 “名字” + “值” 组成的。比如，一个叫做 LOG_LEVEL 的变量可能等于 2；一个叫做 USER_NAME 的变量可能等于 “你的名字”。

环境变量有很多个，PATH 为其中一个。

PATH 指的是，在命令行输入任何指令时，Shell 在电脑的哪些路径中去寻找你所敲的指令。除了 shell builtin 以外，每一个指令都是一个具体的程序（也就是说你可以在系统的各种路径中找到）

每个用户都可以在 /home/你的用户名 下的 .bashrc 文件中找到你的环境变量。不要被后缀名吓到，只是一个txt而已。该文件是一个 Bash 脚本，用来配置环境变量。Bash 是一个简易的脚本语言。该文件是隐藏文件。
```bash
# 你可以通过
ls -a
# 看到这个文件
gedit .bashrc  # 可以打开这个文件
```
你在 Files（Ubuntu自带的文件浏览器）里将“显示隐藏文件”选中也可以。

将你要添加的路径粘贴到文件的最下方就可以了，如：
```bash
export PATH="/home/MyUserName/anaconda3/bin:$PATH"
```
注意，格式一定是 export PATH=“你要添加的路径:$PATH” ！其中，PATH = 是给 PATH 这个变量赋值。：是路径追加。$PATH 是引用 PATH 的值

```Bash
echo $PATH  # 在命令行里，你可以这样将 PATH 的值打印出来
```
那么，当你在命令行里面输入 python 时，Shell 会在 PATH 中寻找叫做 python 的文件。一旦找到，那么就会执行。因为 /home/MyUserName/anaconda3/bin 这个文件夹里面有叫 python 的文件，所以就执行了这个 python。

所以要注意，假如你有多个 python 的版本，一个在 directory1 里面，一个在 directory2 里面
```
export PATH="/directory1:/directory2:$PATH"
```
那么因为先搜索到 directory1，所以就会执行 directory1 里面的 python
```
export PATH="/directory2:$PATH"
export PATH="/directory1:$PATH"
# 这种写法也是一样的，意为先追加 directory2 到 PATH 的前面，再追加 direcotry1 到 PATH 前面。
# 所以最终 directory1 在 directory2 前面。
```
```
export PATH="$PATH:/directory2"
export PATH="$PATH:/directory1"
# 而这样的话，就是执行 direcory2 里面的 python，前提是在原本的 PATH 里面找不到 python。
# 如果在原本的 PATH 里面有 python，那么 Shell 根本不会搜索 directory2。
```

### 5.Virtual Environment 是什么？
virtualenv 是一个 Python 模组/工具，可以通过
```
pip install virtualenv
```
来下载。

它的目的是将 python 的环境配置到 [当前所在文件夹]，这样就不会与系统全局的 python 有任何冲突。

假如你在 dir1 路径下，并且你有 python2 和 python3 两个版本。你可以选择性地，使用其中一个 python 作为虚拟环境。它其实就会将对应 python 的解释器以及你所需要的包全部安装到 dir1 下。这样如果你对这个目录下的 python 做出任何修改，是不会影响到系统全局 python 的。

详情请见视频。

[视频链接]() 尚未录制，敬请期待
