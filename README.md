# How2RunOllamaInCloudStudio
如何在腾讯面向开发者的cloud studio上免费用ollama运行Qwen3-30b-a3b-think Q4量化模型，模型大小19GB

## 1. 在腾讯Cloud Studio上创建一个只有Ubuntu的应用并安装Ollama
[https://cloudstudio.net/my-app](https://cloudstudio.net/my-app)
### 1）创建应用，模板选Ubuntu，应用空间（类似云主机）内存选16C32G及以上CPU主机 或者 8C32G+16G显存T4主机
安装完成后，硬盘使用了~0.5GB
### 2）打开终端，拉取 Ollama 官方镜像（如果未预装）：
docker pull ollama/ollama
安装完成后，硬盘使用了3.6GB

## 2. 运行Ollama，并下载qwen3:30b-a3b-thinking-2507-q4_K_M 模型（从[ollama.com](https://ollama.com/library/qwen3/tags)得到模型的信息和名字）
### 1.1）如果应用空间没有GPU使用下面的命令启动ollama
docker run -d \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --restart always \
  ollama/ollama
此时内存使用了0.8G、硬盘使用了3.6GB
### 1.2）如果应用空间有GPU，直接使用下面的命令启动ollama
docker run -d --gpus=all\
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --restart always \
  ollama/ollama
**如果失败则是因为NVidia显卡在Docker这个Container中运行的关系，用以下方法解决**
Configure NVIDIA Container Runtime to bypass device filtering：A workaround documented in community reports involves telling the NVIDIA runtime to skip strict device cgroup enforcement by editing its config:
Edit or create the config file:

sudo mkdir -p /etc/nvidia-container-runtime
sudo nano /etc/nvidia-container-runtime/config.toml
Add or ensure the following 2 lines are present:

[nvidia-container-cli]
no-cgroups = true
###The key setting is `no-cgroups = true`, which disables cgroup device rule enforcement and avoids the `bpf_prog_query` call .

[nvidia-container-runtime]
debug = "/tmp/nvidia-container-runtime.log"

用^x退出编辑后再次运行ollama启动命令就OK了
sudo systemctl restart docker
docker run -d --gpus=all\
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --restart always \
  ollama/ollama
此时内存使用了0.8G、硬盘使用了3.6GB

### 2）在Docker中下载并启用qwen3:30b-a3b-thinking-2507-q4_K_M 模型
查看容器 ID 或名称
docker ps    
➜  /workspace git:(master) docker ps
**CONTAINER ID**   IMAGE           COMMAND               CREATED          STATUS          PORTS                                           NAMES
**dad073e1a5a7**   ollama/ollama   "/bin/ollama serve"   11 minutes ago   Up 11 minutes   0.0.0.0:11434->11434/tcp, :::11434->11434/tcp   kind_golick
假设容器ID为 dad073e1a5a7(如上面运行的例子），运行下面的命令下载并启用qwen3:30b-a3b-thinking-2507-q4_K_M 模型，下载速度一般为20-40MB/s，该模型19GB大约10~15min完成
docker exec -it dad073e1a5a7 ollama run qwen3:30b-a3b-thinking-2507-q4_K_M --verbose
此时内存使用了~20G、硬盘使用了~22GB，如果有GPU的话GPU显存使用了18G（T4的话只有16G都占满），GPU占用率80%

**实测qwen3:30b-a3b-thinking-2507-q4_K_M 模型速度**
1）16C32G CPU应用空间 达到17token/s！（该配置每天能薅1小时）
2）8C32G + 16G显存T4的GPU应用空间 达到30tokens/s（该配置每周能薅10+小时）
3）20C116G + 24G显存A10的GPU应用空间 达到100tokens/s！！！（该配置每周能薅3+小时）


