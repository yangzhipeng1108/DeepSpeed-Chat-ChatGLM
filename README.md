
单GPU运行

actor-model运行过程中：单A100使用deepspeed运行hatglm-6b会OOM,新增quantization,解决OOM问题
python train.py --actor-model chatglm-6b  --reward-model facebook/opt-350m --deployment-type single_gpu

单节点运行

python train.py --actor-model chatglm-6b  --reward-model facebook/opt-350m --deployment-type single_node
