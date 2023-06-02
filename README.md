
# 单GPU运行

actor-model运行过程中：单A100使用deepspeed运行chatglm-6b会OOM,新增quantization,解决OOM问题

python train.py --actor-model chatglm-6b  --reward-model facebook/opt-350m --deployment-type single_gpu

chatglm-6b和opt-350m 因为tokenizer不同 会出现冲突

python train.py --actor-model chatglm-6b  --reward-model chatglm-6b --deployment-type single_gpu

quantization 8 之后 step3 不同模块会抢占launch地址

# 单节点运行

python train.py --actor-model chatglm-6b  --reward-model chatglm-6b --deployment-type single_node

最少需要5张A100或者A800

可以修改成deepspeed step 3
DeepSpeed-Chat-ChatGLM/training/step3_rlhf_finetuning/training_scripts/single_node
/run_chatglm-6b.sh  


# 推理 inference

sh inference.sh


# 网页Demo

## Gradio

基于Gradio的网页Demo，您可以运行本仓库中的web_demo.py：

python web_demo.py
