## 打包说明

#### macos M1 Max 

###### Python版本：3.10 arm64 （非虚拟环境）

###### 包依赖：

```python
certifi               2023.7.22
charset-normalizer    3.3.2
idna                  3.4
Nuitka                1.8.6
numpy                 1.25.2
opencv-python         4.6.0.66
ordered-set           4.1.0
Pillow                10.1.0
pip                   23.3.1
PyQt5                 5.15.7
PyQt5-sip             12.11.0
requests              2.31.0
setuptools            65.5.0
torch                 1.13.0
typing_extensions     4.8.0
urllib3               2.0.7
wheel                 0.41.3
zstandard             0.22.0
```

###### 打包命令：

```python
python -m nuitka \
          --standalone \
          --onefile\
          --enable-plugin=pyqt5,numpy\
          --macos-create-app-bundle\
          --output-dir=build\
          --assume-yes-for-download\
          --macos-app-version=1.0\
          --disable-console\
          --include-data-dir=./static/=./static/\
					--macos-app-icon=./app.ico\
          main.py

```

#### Win10 64

###### 打包命令：

```python
nuitka --standalone --lto=no --report=report.xml --mingw64 --show-progress --show-memory --enable-plugin=pyqt5 --plugin-enable=torch --plugin-enable=numpy --output-dir=dist --include-data-dir=./static/=./static/ --windows-icon-from-ico=app.ico main.py
```

