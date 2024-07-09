## 打包说明

#### MacOS AppleSilicon

###### Python版本：3.10.11 （非虚拟环境）

###### Mac版本：13.4 M1 Max 32+1T

###### 包依赖：

```python
certifi==2023.7.22
charset-normalizer==3.3.2
idna==3.4
Nuitka==1.8.6
numpy==1.25.2
opencv-python==4.6.0.66
ordered-set==4.1.0
Pillow==10.1.0
PyQt5==5.15.7
PyQt5-sip==12.11.0
requests==2.31.0
torch==1.13.0
typing_extensions==4.8.0
urllib3==2.0.7
wheel==0.41.3
zstandard==0.22.0
```

###### 打包命令：

```shell
python -m nuitka \
          --standalone \
          --onefile\
          --enable-plugin=pyqt5\
          --macos-create-app-bundle\
          --output-dir=build\
          --assume-yes-for-download\
          --macos-app-version=2.0\
          --disable-console\
          --include-data-dir=./static/=./static/\
					--macos-app-icon=./app.icns\
					--include-data-files=./model.py=./model.py FilterSimulation.py
```

注意：

1. 保证系统架构为arm64：使用arch命令查看；切换arm：**arch -arm64 zsh**  切换x86：**arch -x86_64 zsh**

2. 确保下载的第三方库均为arm64版本

3. 打包过程中发现若代码中存在**读取yaml文件**，打包后的**app无法双击打开**且不报错（可以在termial中打开）；替换为**json文件**后正常打开。

4. 打包后的 **info.plist文件** 和 **Resource文件夹** 需要手动移动到 **app的Contents**目录下面

5. 关于PyQt5：使用pip安装的qt5虽然为arm64版本，但是存在如下问题：

   ```python
   ImportError: dlopen(/Users/User/environments/env/lib/python3.11/site-packages/PyQt5/QtCore.abi3.so, 0x0002): Symbol not found: __ZTVNSt3__13pmr25monotonic_buffer_resourceE 
       Referenced from: <645DDC2C-F655-324A-BE87-40804F3AC471> /Users/User/environments/env/lib/python3.11/site-packages/PyQt5/Qt5/lib/QtCore.framework/Versions/5/QtCore
       Expected in:     <54E8FBE1-DF0D-33A2-B8FA-356565C12929> /usr/lib/libc++.1.dylib
   ```

   解决方案如下：

   ```python
   # 使用homebrew安装pyqt5
   brew install PyQt5
   ```

   ```python
   # 拷贝文件
   cp /opt/homebrew/Cellar/pyqt@5/5.15.7_1/lib/python3.10/site-packages/PyQt5 /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-package/PyQt5
   ```

#### MacOS Intel

###### Python版本：3.9 （非虚拟环境）

###### Mac版本：11.3 Inter i5 16+512G

###### 包依赖：

```python
certifi==2023.7.22
charset-normalizer==3.3.2
idna==3.4
Nuitka==1.8.6
numpy==1.25.2
opencv-python==4.6.0.66
ordered-set==4.1.0
Pillow==10.1.0
PyQt5==5.15.2
PyQt5-sip==12.11.0
requests==2.31.0
torch==1.13.0
typing_extensions==4.8.0
urllib3==2.0.7
wheel==0.41.3
zstandard==0.22.0
```

###### 打包命令：同上

#### Win11 64

###### Python版本：3.10.11 （非虚拟环境）

###### 包依赖：

```python
Nuitka==1.8.5
numpy==1.25.2
opencv-python==4.6.0.66
ordered-set==4.1.0
Pillow==10.1.0
PyQt5==5.15.7
PyQt5-Qt5==5.15.2
PyQt5-sip==12.13.0
torch==1.13.0
typing_extensions==4.8.0
zstandard==0.22.0
```

###### 打包命令：

```python
nuitka 
--standalone 
--lto=no 
--report=report.xml 
--mingw64 
--show-progress 
--show-memory 
--enable-plugin=pyqt5 
--plugin-enable=torch 
--plugin-enable=numpy 
--output-dir=dist 
--include-data-dir=./static/=./static/ 
--windows-icon-from-ico=app.ico 
main.py
```

注意：

1. 需要安装mingw64： https://link.zhihu.com/?target=https%3A//winlibs.com/   （GCC 13.2.0）
