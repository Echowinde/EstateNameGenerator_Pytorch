import numpy as np
import torch
from torch.autograd import Variable
import os
import sys
sys.path.append('./src/')
from model import Config


def load_model_and_vocabulary():
    """
    加载训练好的模型及词库并返回
    """
    if not os.path.exists('./model.pth'):
        print("请先运行train.py训练模型")

    model = torch.load('./model.pth')
    vocabulary = np.load('./data/vocabulary.npz', allow_pickle=True)
    word2ix = vocabulary['word2ix'].item()
    ix2word = vocabulary['ix2word'].item()
    return model, word2ix, ix2word


def generator(model, ix2word, word2ix, config, start=''):
    """
    根据输入生成楼盘名
    """
    if start != '':  # 如果输入了起始字
        results = list(start)
        start_len = len(results)
        inputs = Variable(torch.Tensor([word2ix[results[0]]]).view(1, 1).long())
        hidden = None
    else:  # 输入为空，随机生成楼盘名
        results = []
        start_len = 0
        inputs = Variable(torch.Tensor([word2ix["<SOS>"]]).view(1, 1).long())  # input设置为<SOS>
        hidden = (Variable(torch.rand(2, 1, config.hidden_dim)), Variable(torch.rand(2, 1, config.hidden_dim)))  # 随机初始化

    for i in range(10):
        output, hidden = model(inputs, hidden)

        if i < start_len:  # 名字仍在起始字范围内
            word = start[i]  # 模型输入直接设置为输入的起始字，目标是获取起始字最后的hidden状态
            inputs = Variable(inputs.data.new([word2ix[word]])).view(1, 1)
        else:
            top = output.data[0].topk(1)[1][0].item()  # 概率第一的结果
            word = ix2word[top]
            results.append(word)
            inputs = Variable(inputs.data.new([top])).view(1, 1)

        if (word == "<EOS>") | (word == "<\\s>"):  # 输出结束
            break

    return ''.join(results[:-1])


if __name__ == "__main__":
    model, word2ix, ix2word = load_model_and_vocabulary()
    config = Config()
    while True:
        start_word = input("请输入任意个起始字或随机生成楼盘名，输入exit退出：")
        if start_word == 'exit':
            break
        result = generator(model, ix2word, word2ix, config, start_word)
        print("楼盘名： ", result)
