# 运行说明
## 选择系统镜像
ubuntu 20.04+

## 编译
参考文档 https://mortal.ekyu.moe/user/build.html

```sh
git clone https://github.com/Equim-chan/Mortal.git
cd Mortal
conda env create -f environment.yml
conda activate mortal
pip3 install torch

apt install cargo
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
```

## 配置模型
1. 模型文件下载地址 https://share.weiyun.com/XVEK25Wp 密码:bevnbk
2. 模型上传到服务器上，如/root/Mortal/mortal.pth
3. 创建配置文件 mortal/config.toml,内容如下:
```yaml
[control]
state_file = '/root/Mortal/mortal.pth'
version = 4

[resnet]
conv_channels = 192
num_blocks = 40
enable_bn = true
bn_momentum = 0.99
```
说明：state_file 的路径箐根据位置更新

## 运行
 1. 安装依赖
```sh
pip install fastapi pydantic 
apt install unicorn
```
2. 启动fastapi inference server
```
cd mortal
uvicorn mortal_api_server:app --workers 8
```
3. 压测
```
ab -n 40 -c 8 -p post_data.json -T "application/json" http://localhost:8000/infer/
```