

# Description
This is the implementation of Qingxing Cao et al.'s CVPR-17 work [Attention-Aware Face Hallucination via Deep Reinforcement Learning](https://arxiv.org/abs/1708.03132). If you have any questions or suggestions of this work, please feel free to contact the authors by sending email to 'caoqxATmail2.sysu.edu.cn' or 'shiyk3ATmail2.sysu.edu.cn'.

# Citation
Please cite Attention-FH in your publications if it helps your research:
```
@article{Cao2017Attention,
  title={Attention-Aware Face Hallucination via Deep Reinforcement Learning},
  author={Cao, Qingxing and Lin, Liang and Shi, Yukai and Liang, Xiaodan and Li, Guanbin},
  pages={1656-1664},
  year={2017},
}

```

# Prerequisites
- Computer with Linux or OSX
- Torch-7
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.
- Cuda 8.0

# Installing dependencies
- Install Torch:  http://torch.ch/docs/getting-started.html
- Install Element-Research/rnn: https://github.com/Element-Research/rnn

# Train
- To put your own training and testing files as:
```
../lfw_funneled_dev_128/train/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
../lfw_funneled_dev_128/test/Aaron_Guiel/Aaron_Guiel_0001.jpg

```

Note that our data has processed with 2 points aligned and crop the centric part. We suggest you to use [CFSS](https://github.com/zhusz/CVPR15-CFSS) or other face alignment methods to pre-process your data. Then you can train the model by use the following command:
```
python RNN_main.lua
```
