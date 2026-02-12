vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --host 0.0.0.0 \
    --port 8081 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto