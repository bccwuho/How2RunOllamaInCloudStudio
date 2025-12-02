# 结论
- **vLLM使用的一些优化手段需要A系列的GPU，所以A10和L40可以，V100和T4已经不能支持了;而Ollama上即使无GPU也能用**
- **Qwen3-30B-A3B-Thinking-2507-AWQ-4bit在Ollama@A10上的速度能到100t/s、单并发且API无法配key；vLLM速度能到200t/s、不排队能6并发吞吐量达到300-400t/s且API能配单个key了，相当优秀，但24G显存的A10只能跑16K上下文**
- 🔴**Qwen3-30B-A3B-Thinking-2507-FP8实测在vLLM@48C196G内存48G显存L40上单发速度能到100t/s、不排队能6并发吞吐量达到240/s且API能配单个key了，且上下文能到100K，性能相当优秀!!!**
- vLLM上目前CloudStudio能跑的最好模型是智力4.35的Qwen3-30B-A3B-Thinking-2507-FP8 和 智力4.3的Qwen3-30B-A3B-Thinking-2507-AWQ-4bit

## 0.腾讯云CloudStudio的CUDA驱动 和 Docker 和 NVIDIA Container Toolkit都已经装好，但如果遇到类似下面cgroup问题，例如失败报类似下面的错误（本质是nVidia在docker中运行错，要打开一些权限）按一下方法解决即可
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy' <BR>
nvidia-container-cli: mount error: failed to add device rules: unable to find any existing device filters attached to the cgroup: bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not  <BR>permitted: unknown. <BR>

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
**若能正常显示 A10或L40 与CUDA驱动版本，即表示容器可用 GPU** <BR>

```bash
sudo mkdir -p /etc/nvidia-container-runtime
sudo nano /etc/nvidia-container-runtime/config.toml

Add or ensure the following 2 lines are present:

[nvidia-container-cli]
no-cgroups = true
###The key setting is no-cgroups = true, which disables cgroup device rule enforcement and avoids the bpf_prog_query call .

[nvidia-container-runtime]
debug = "/tmp/nvidia-container-runtime.log"

用^x（按CTRL+X）退出编辑后再次运行ollama启动命令就OK了
```

## 1. vLLM运行Qwen3-30B-A3B-Thinking-2507-AWQ-4bit量化模型，提供API Web接口给CherryStudio使用 <BR>
1）20C116G内存24G显存A10的GPU速度最大可达200t/s，6并发吞吐量达到峰值300~400t/s（并发数>6后就开始排队等待），GPU用到97%显存用到21.7/22.5但CPU和内存基本都是空闲，50GB硬盘用了53GB但重启后 
   还是可以直接用的（应该是模型文件+vLLM23GB没有超50GB，加上一些临时文件超了），该环境下的极限值最大上下文长度16384和GPU显存利用率0.9  <BR>
2）8C140G内存32G显存V100的GPU 由于GPU太老计算能力 7.0，跑不了AWQ模型（计算能力至少为 8.0SM80），所以V100机器建议用GPTQ的量化模型 <BR>

### 1.1 安装方法：
应用空间选Ubuntu + Docker安装vLLMhttps://github.com/bccwuho/How2RunOllamaInCloudStudio/blob/main/How2RunVLLMInCloudStudio.md  <BR>
API接口：https://2d255b6bdde54e2996aa98333d5bc10d--8000.ap-shanghai2.cloudstudio.club/v1/   <BR>
======完整安装过程如下====<BR>
```bash
export HF_HOME=~/.cache/huggingface   
mkdir -p "$HF_HOME"                  
docker pull vllm/vllm-openai:latest    
```

### 1.2 重启后用下面命令启动模型！api-key用多个sk-key1,sk-key2测试失败只能一个key！24GA10显卡用这个AWQ量化模型
```bash
export HF_HOME=~/.cache/huggingface                  
docker run --rm --gpus all --ipc=host \               
  -p 8000:8000 \                                    
  -v "$HF_HOME":/root/.cache/huggingface \         
  vllm/vllm-openai:latest \                       
  --model cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit \      
  --api-key sk-123 \                                 
  --dtype auto \                                    
  --max-model-len 16384 \                           
  --gpu-memory-utilization 0.90
```
    
== 重启后用下面命令启动模型！api-key用多个sk-key1,sk-key2测试失败只能一个key！32GV100显卡用这个，但目前还没有启动成功过？？？   <BR>
```bash
export HF_HOME=~/.cache/huggingface               
export TORCHDYNAMO_DISABLE=1                     
export VLLM_DISABLE_FA2=true                     

docker run --rm --gpus all --ipc=host \          
  -p 8000:8000 \                                  
  -v "$HF_HOME":/root/.cache/huggingface \          
  vllm/vllm-openai:latest \                        
  --model btbtyler09/Qwen3-30B-A3B-Thinking-2507-gptq-4bit \       
  --api-key sk-123 \                                  
  --dtype auto \                                        
  --max-model-len 8192 \                                  
  --gpu-memory-utilization 0.85 \                         
  --reasoning-parser qwen3                            <
```

### 1.3 测试
```bash
curl http://localhost:8000/v1/chat/completions \       
  -H "Content-Type: application/json" \                
  -d '{                                                 
    "model": "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit",          
    "messages": [{"role":"user","content":"用中文一步步思考：24 * 17 等于多少？"}],       
    "stream": false                                     
  }'    <BR>
```

## 2. Qwen3-30B-A3B-Thinking-2507-FP8运行在vLLM@48C196G内存48G显存（实际45G）L40
### 2.1 一个命令安装vLLM + 加载Qwen3-30B-A3B-Thinking-2507-FP8模型（模型31GB最大上下文100K，KV Cache大致11G把45G显存基本用足了！） 和 启动API服务
**纯保留容器（最省事；不映射卷）, 重启后不会重下，但如果以后你 docker rm 了容器，缓存就跟着没了。**
```bash
docker run --name vllm-qwen -d --gpus all \
  --ipc=host \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-model-len 100000 \
  --api-key sk-123
```

**之后用下面命令**
docker stop vllm-qwen               # 停止
docker start vllm-qwen              # 后台启动
docker logs -f vllm-qwen            # 看日志
**想要前台看日志启动：docker start -ai vllm-qwen**


### 2.2 本地测试
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-123" \
  -d '{
        "model": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",   \
        "messages": [{"role":"user","content":"你好，请用一句话自我介绍"}]   \
      }'
```

