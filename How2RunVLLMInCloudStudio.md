# 结论
- **vLLM使用的一些优化手段需要A系列的GPU，所以A10和L40可以，V100和T4已经不能支持了;而Ollama上即使无GPU也能用**
- **Qwen3-30B-A3B-Thinking-2507-AWQ-4bit在Ollama@A10上的速度能到100t/s、单并发且API无法配key；vLLM速度能到200t/s、不排队能6并发吞吐量达到300-400t/s且API能配单个key了，相当优秀，但24G显存的A10只能跑16K上下文**
- vLLM上目前CloudStudio能跑的最好模型是智力4.3的Qwen3-30B-A3B-Thinking-2507-AWQ-4bit，

vLLM运行Qwen3-30B-A3B-Thinking-2507-AWQ-4bit量化模型，提供API Web接口给CherryStudio使用
1）20C116G内存24G显存A10的GPU速度最大可达200t/s，6并发吞吐量达到峰值300~400t/s（并发数>6后就开始排队等待），GPU用到97%显存用到21.7/22.5但CPU和内存基本都是空闲，50GB硬盘用了53GB但重启后
   还是可以直接用的（应该是模型文件+vLLM23GB没有超50GB，加上一些临时文件超了），该环境下的极限值最大上下文长度16384和GPU显存利用率0.9
2）8C140G内存32G显存V100的GPU 由于GPU太老计算能力 7.0，跑不了AWQ模型（计算能力至少为 8.0SM80），所以V100机器建议用GPTQ的量化模型

# 安装方法：
应用空间选Ubuntu + Docker安装vLLMhttps://github.com/bccwuho/How2RunOllamaInCloudStudio/blob/main/How2RunVLLMInCloudStudio.md  <BR>
API接口：https://2d255b6bdde54e2996aa98333d5bc10d--8000.ap-shanghai2.cloudstudio.club/v1/   <BR>
======完整安装过程如下====<BR>
export HF_HOME=~/.cache/huggingface   <BR>
mkdir -p "$HF_HOME"                  <BR>
docker pull vllm/vllm-openai:latest    <BR>

# 重启后用下面命令启动模型！api-key用多个sk-key1,sk-key2测试失败只能一个key！24GA10显卡用这个AWQ量化模型
export HF_HOME=~/.cache/huggingface                  <BR>
docker run --rm --gpus all --ipc=host \               <BR>
  -p 8000:8000 \                                    <BR>
  -v "$HF_HOME":/root/.cache/huggingface \         <BR>
  vllm/vllm-openai:latest \                        <BR>
  --model cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit \      <BR>
  --api-key sk-123 \                                 <BR>
  --dtype auto \                                    <BR>
  --max-model-len 16384 \                           <BR>
  --gpu-memory-utilization 0.90                     <BR>
== 重启后用下面命令启动模型！api-key用多个sk-key1,sk-key2测试失败只能一个key！32GV100显卡用这个，但目前还没有启动成功过？？？   <BR>
export HF_HOME=~/.cache/huggingface               <BR>
export TORCHDYNAMO_DISABLE=1                     <BR>
export VLLM_DISABLE_FA2=true                     <BR>

docker run --rm --gpus all --ipc=host \          <BR>
  -p 8000:8000 \                                  <BR>
  -v "$HF_HOME":/root/.cache/huggingface \          <BR>
  vllm/vllm-openai:latest \                         <BR>
  --model btbtyler09/Qwen3-30B-A3B-Thinking-2507-gptq-4bit \       <BR>
  --api-key sk-123 \                                  <BR>
  --dtype auto \                                        <BR>
  --max-model-len 8192 \                                  <BR>
  --gpu-memory-utilization 0.85 \                         <BR>
  --reasoning-parser qwen3                            <BR>
====
如果失败报类似下面的错误（本质是nVidia在docker中运行错，要打开一些权限）见https://github.com/bccwuho/How2RunOllamaInCloudStudio 1.2的解决方法    <BR>
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy' <BR>
nvidia-container-cli: mount error: failed to add device rules: unable to find any existing device filters attached to the cgroup: bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not  <BR>permitted: unknown. <BR>

====
curl http://localhost:8000/v1/chat/completions \       <BR>
  -H "Content-Type: application/json" \                <BR>
  -d '{                                                 <BR>
    "model": "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit",          <BR>
    "messages": [{"role":"user","content":"用中文一步步思考：24 * 17 等于多少？"}],       <BR>
    "stream": false                                     <BR>
  }'    <BR>
