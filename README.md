
单GPU运行

python train.py --actor-model chatglm-6b  --reward-model facebook/opt-350m --deployment-type single_gpu

单节点运行

python train.py --actor-model chatglm-6b  --reward-model facebook/opt-350m --deployment-type single_node
