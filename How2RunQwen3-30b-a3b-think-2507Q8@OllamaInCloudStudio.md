# How to Run Qwen3-30b-a3b-think-2507Q8@Ollama In CloudStudio

## 0. 先确保主机已按官方文档完成 NVIDIA 容器nvidia-container-toolkit运行时配置
参考https://docs.ollama.com/docker

## 0.1 Install with Apt
### 0.1.1 Configure the repository 
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \ <BR>
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg <BR>
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \ <BR>
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \ <BR>
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list <BR>
sudo apt-get update <BR>

### 0.1.2 Install the NVIDIA Container Toolkit packages
sudo apt-get install -y nvidia-container-toolkit <BR>

## 0.2 Configure Docker to use Nvidia driver
sudo nvidia-ctk runtime configure --runtime=docker <BR>
sudo systemctl restart docker   ### 这步在CVM的docker环境下非必须，只要cat /etc/docker/daemon.json 有nvidia-container-runtime字样就行<BR>

## 1. 安装并在后台运行ollama
docker run -d --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  -e OLLAMA_DEBUG=1 \
  -e OLLAMA_NO_MMAP=1 \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_GPU_OVERHEAD=1073741824 \
  --restart always \
  ollama/ollama:latest

Tips：<BR>
1、可以用docker stop ollma 和 docker rm ollama 来停止和删除这个ollama container！然后docker start ollama重启<BR>
2、各个参数的解释<BR>
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \    ###这句在CVM下失败，去掉即可，但不知对锁定权重在内存是否能成功？？？<BR>
  -- e OLLAMA_FLASH_ATTENTION=1 \ ###似乎最新版默认已用Flash Attention这种现代LLM技术（速度X2；KV 显存➗2？）<BR>
  -e OLLAMA_GPU_OVERHEAD=6442450944 \ 预留6G显存已经很稳定了；但后来咨询AI发现这个OVERHEAD不会用来预留给KV，所以Linux一般先不设（默认 0）或设 0.5–1 GiB就够了；只有你还要给其它进程留显存，或在 Windows/WDDM 上跑，才考虑到 2 GiB 左右。GitHub 里也有维护者用 **512 MiB（536,870,912 字节）**作为示例预留值。​<BR>
  -e OLLAMA_KV_CACHE_TYPE=f16 \  默认KV Cache的是f16=fp16，实际设为q8_0更省显存且性能几无损失<BR>

**3、遇到类似下面cgroup问题就** <BR>
==docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: mount error: failed to add device rules: unable to find any existing device filters attached to the cgroup: bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not permitted: unknown.==<BR>

sudo mkdir -p /etc/nvidia-container-runtime <BR>
sudo nano /etc/nvidia-container-runtime/config.toml <BR>

Add or ensure the following 2 lines are present: <BR>

[nvidia-container-cli] <BR>
no-cgroups = true <BR>
###The key setting is no-cgroups = true, which disables cgroup device rule enforcement and avoids the bpf_prog_query call . <BR>

[nvidia-container-runtime] <BR>
debug = "/tmp/nvidia-container-runtime.log" <BR>

**用^x（按CTRL+X）退出编辑后再次运行ollama启动命令就OK了** <BR>

## 2. 下载并运行模型qwen3:30b-a3b-thinking-2507-q8_0
## 2.1 下载模型
**docker exec -it ollama ollama pull qwen3:30b-a3b-thinking-2507-q8_0** <BR>

## 2.2 将自己希望的参数写入 Modelfile ,并创建模型别名（如果GPU显存够则Optional）
cat <<'EOF' | docker exec -i ollama sh -lc 'cat > /root/Modelfile'
FROM qwen3:30b-a3b-thinking-2507-q8_0

PARAMETER num_ctx 32768         
PARAMETER num_predict 20480    
PARAMETER num_keep -1           
PARAMETER use_mlock 1           
PARAMETER use_mmap 0
EOF

docker exec  ollama cat /root/Modelfile #来检查文件是否写入正确

Tips：参数说明如下  <BR>
PARAMETER num_ctx 32768         # 32K 上下文<BR>
PARAMETER num_predict 15360     # 最多一次性输出 \~15K，原来设16K，后来发现Cherry Stuidio16K输出截断+预留16K下文再+最初的Prompt会导致超32K，会KV cache清空，-5问题继续就会prefill超时！所以改成15K避免之；再后来太短也不行，-5这道题有时会思考15-20K，思考中截断cherry studio不会把思考中的tokens变成上下文从而又要重新开始做题了；**所以按这个说法（KVcache会清空）应该设为20K（32K-12K上文Prefill最大数见后面的阐述），这样在32K上下文能解决的范围内，最大解决的问题输入是12K，并尽可能1次做对（最长输出20K，对应推理时间\~30min）！！！**Prefill 的计算复杂度是 O(N²)（主要来自 Attention 的 QK^T 计算），而 Decode 是 O(N)（每次只算一个 token 对全部历史的 attention）。因此在 8–12K 这类长度时看到的 Prefill/Decode 比值常常只有 \~1–3×；只有在短上下文或强批处理时，Prefill 才更容易达到 10× 以上最高100x。<BR>
PARAMETER num_keep -1           # 可能是默认，最大限度保留历史到 KV，但仍然不超过 num_ctx 上下文 <BR>
PARAMETER num_keep 0            # 不保留历史到 KV，旧消息在 num_ctx 内重算 KV；如果设成0，则如果历史消息很长时Prefill就有可能超时（CherryStudio是5min，在<几何题-5第一次输出16K后再继续时就会超时失败导致历史消息>=16K的对话进行不下去，并且挂住系统一段时间影响其他对话直到这个Prefill结束大约20min左右）<BR>
PARAMETER use_mlock 1           # 锁定权重在内存，避免 pageout 抖动， warning: parameter use_mlock is deprecated<BR>
PARAMETER use_mmap 0            # 进一步禁用 mmap（配合上面 OLLAMA_NO_MMAP，模型参数整包加载减少IO抖动）<BR>
PARAMETER num_gpu 999  	        # Ollama会尝试把尽可能多的层放到 GPU/ num_gpu是加载到GPU的层数，比ollama默认的速度更快一点但占用VRAM更多更激进一点？？？但也报错了，放弃该参数<BR>

**用刚做好的Modelfile创建带参数的模型别名**<BR>
docker exec -it ollama ollama create qwen3-30b-a3b-2507-Q8:ctx32k-mlock -f /root/Modelfile

## 2.3 运行模型
### 2.3.1 交互运行（CLI）
**docker exec -it ollama ollama run qwen3-30b-a3b-2507-Q8:ctx32k-mlock --verbose**

Tips：<BR>
**用docker logs -f ollama | grep -E 'memory\.|overhead|offload to cuda' 看KV在GPU和CPU中如何分配的（22.5G显存的A10）：** <BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:206 msg="model weights" device=CUDA0 size="18.5 GiB"<BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:211 msg="model weights" device=CPU size="11.7 GiB"<BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:217 msg="kv cache" device=CUDA0 size="1.9 GiB"<BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:222 msg="kv cache" device=CPU size="1.1 GiB"<BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:228 msg="compute graph" device=CUDA0 size="190.5 MiB"<BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:233 msg="compute graph" device=CPU size="84.0 MiB"<BR>
time=2025-10-22T02:17:58.181Z level=DEBUG source=device.go:238 msg="total memory" size="33.5 GiB"<BR>

```
之前在22.5G显存的A10和Ollama缺省参数下，运行qwen3-30b-a3b-2507-Q8模型，在长上下文问题时（例如几何题-5，回答在17-42K时）会API crash（无反应，要靠等待20min以上或重启Docker来恢复），所以我总结一下我的需求，请AI给出相关的docker run命令：
1、在24G显存（实际22.5G）的A10显卡20核116G内存的机器上要运行ollama，用qwen3-30b-a3b-2507-Q8模型，模型大小32GB
2、希望留6G显存给KV Cache（对应KV在FP16下能预留64K的上下文）
3、num_keep设为0，保证以前的对话不保留在KV，之前的对话在下面num_ctx长度内会被重新计算KV（这个不保留KV对多个对话是最小化KV的结果）
4、num_ctx设为32K，32K上下文的对话，最大会产生3G显存需求（KV cache用默认fp16的情况下）
5、num_predict设为15K，一次性最大输出达到\~15K时会截断
6、使用mlock把模型权重文件锁定在内存中，避免Pageout（在内存有压力时例如内存占用达80-85%时）产生IO抖动造成API的crash（无反应，要靠等待20min以上或重启Docker来恢复）
7、禁用mmap模型参数整包加载减少IO抖动，避免crash
但是最后发现Crash的核心原因是：
1）因为没设置OVERHEAD在其他耗显存时（非模型权重+KV cache）突然爆显存导致Crash（发现过1次）；这个主要靠设置OLLAMA_GPU_OVERHEAD来解决
同时即使预留了显存给KV Cache，也会在多对话且长上下文时超过原来的预留，这样显存会不够，就会有显存和内存间传输，就有可能会导致crash；这个靠OLLAMA_KV_CACHE_TYPE=q8_0+调节num_ctx/num_predict/num_gpu来解决
2）模型参数不常驻显存+内存，在内存紧张时（>=85%）或只要是默认启用mmap时就有可能与硬盘Pageout时会IO抖动就有可能会导致crash；这个主要靠内存足够大和禁用mmap来解决
3）最多的情况：首字节的Prefill阶段过长Timeout后导致Crash（Prefill超5分钟后发送给Cherry Studio端，但Cherry Studio已经timeout了所以ollama进程就锁死在那里），通过num_keep=-1使得历史消息的KV Cache最大化加速，但当前轮的Prompt与历史消息之间也要计算KV，此时如果历史消息过长且速度过慢就可能5minTimeout，目前OLLAMA_KV_CACHE_TYPE=f16时实测8K历史消息+继续OK，16K历史消息+199tokens Prompt卡壳；这个靠OLLAMA_KV_CACHE_TYPE=q8_0提高速度来解决，实测KV=q8_0时KV cache 1G在GPU0.5G在CPU，上文最长可达12-13K才5minTimeout\~a²=（12k）²KV计算量，如果始终在一个对话里靠KV cache的话即使达到32K上下文在输出16K后+"继续"也OK了因为只有~2ab=（16K*2*2）计算量（其中a=历史消息tokens数，b=当前prompt的tokens数）
```

#### 2.3.2 或者走 HTTP API（默认 11434）来测试
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-30b-a3b-2507-Q8:ctx32k-mlock",
  "prompt": "hello",
  "stream": false
}'

# 3. 最终结论：
🔴1）在24GB显存20C116G内存的A10机器上GPU显存不足容纳qwen3-30b-a3b2507Q8（**完成诸如翻译/辅导高中生作业/中级程序员之类的工作（智力~GPT4.35接近DS R1满血老版本））的32GB模型需要GPU/CPU混合推理的情况下，Prefill是主要瓶颈**（KV设q8_0时实测推理生成速度为30t/s，Prefill速度受混合推理拖累实测40t/s）<BR>
🔴2）**业务上Ollama只能为单人服务且在Ollama CLI上体验超好，实测过使用2hour以上完全正常；但用Cherry Studio客户端时体验受限但也能具备在一定上下文条件下**<BR>
具体体验受限如下：<BR>
1）**在Cherry Studio中首次最大问题输入控制在12K，上下文可以达到32K最大输出20K时保持在一个对话中**（防止KV cache切换对话时clear）持续问完体验稳定（长对话时保持耐心且不要中断，例如几何题-5全对，回答一般在12-20K需要大致15-30min）；当Prefill 5min超时或被中断时系统对后面的对话无反应，此时restart docker后重新加载模型即可<BR>
2）在ollama CLI下没有上下文的限制（哪怕模型文件中设了num_ctx=32K）也不会Prefill Timeout（似乎KV Cache一直可以扩展只要内存够），实际对话超2hours，我的测试题目走了近2遍，一切都还正常，体验非常好除了MD格式没有render外<BR>
3）32G的V100 Q8和Q4实测速度都只有A10机器的80%，分别是Q4 80t/s，Q8 24t/s（虽绝大部分都在GPU上了但V100机器CPU是8C比A10的20C弱不少），且比A10还贵，所以一般不用！<BR>
🔴4）在22.5G显存的A10的Ollama下运行qwen3-30b-a3b-2507-Q8模型最佳参数：<BR>
**1、OLLAMA_GPU_OVERHEAD 为1GB<BR>**
**2、OLLAMA_KV_CACHE_TYPE=q8_0  ；减小KV cache占用VRAM，加速Prefill速度（实测5minTimeout时间够Prefill最大上文可以12-14K）<BR>**
**3、启用OLLAMA_FLASH_ATTENTION（可能是默认）<BR>**
4、禁用mmap<BR>
**5、Optional：PARAMETER num_ctx 32768**        # 32K 上下文 和 PARAMETER num_predict 20480     # 最多一次性输出 20K<BR>
6、这样在目标32K上下文问题（最大输出20K的）,KV cache不考虑flashAttention（可能是默认）和默认KV=fp16的情况下最大预留是3GB（部分在1.9G在显存1.1G在内存，约等于模型权重在VRAM和RAM上的比例）；**KV=q8_0时最大预留是1.5GB，1G在显存，0.5G在内存**<BR>
🔴5）**现在本地跑LLM，最重要的是显存够加载完整的权重文件+预留KV Cache（32Kctx默认F16对于Qwen3-30b-a3b\~3GB）；其次才是GPU，主流4070算力足够但16G显存不够，所以想用得好还要上24G4090或32G5090，成本依旧高；32G-128G大内存AMD8745（核显780接近1060）/AI-MAX 390（核显接近4060）可能是性价比高一点的替代**<BR>
🔴6）**现阶段（2025/10）消费级GPU机器上最佳选择：从业务上跑Qwen3-30b-a3b-think-AWQ4模型（智力GPT4.3，与新版DS V3相当）@vLLM（擅长纯GPU）最合适6并发吞吐量能达200-300t/s；硬要上Qwen3-30b-a3b-Q8的话只能勉强Ollama一个人玩且上下文32KAI-MAX390可能达到40t/s？？？极限方案可能是15K元的128GAI-MAX390跑Qwen3-30b-a3b-FP16（与老版DS R1相当）@vLLM单人达到40t/s？？？**<BR>
