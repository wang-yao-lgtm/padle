# padle
brain-inspired-replay结果可重复性验证
以及网络改进，将基于vae的生成网络替换成conterfact网络，
原论文链接：
https://github.com/GMvandeVen/brain-inspired-replay
github.com:anonymous-user-256/mlrc-cgn

# 改进模型与原模型极其相似，主体相同，生成网络由变分自编码换成gan生成网络，且生成目标数据不一致，详情可参见原论文，
brain-inspired-replay网络性能良好，实验结果可重复，个人改进网络易发生过拟合(可能不是这原因)，测试集数据效果极差，初步分析应当是gan在训练过程中生成数据质量过差导致,
