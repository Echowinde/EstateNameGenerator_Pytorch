import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
from model import *


def prepare_data(house_data, col='alias'):
    """
    处理数据，生成数据数组和word-index/index-word字典作为词库
    """
    data = []
    name_list = house_data[col].values  # 默认取alias列的数据（去除了地产商和城市前缀，若不需要去除可取name列）
    for house in name_list:
        house = list(house)
        house.append("<EOS>")
        house.insert(0, "<SOS>")
        if len(house) < 10:  # 不足10的数据补齐
            for i in range(len(house), 10):
                house.insert(i, "<\\s>")
        data.append(house)

    # 词库和字典生成
    words = [word for house in data for word in house]
    words = set(words)
    word2ix = dict((c, i) for i, c in enumerate(words))
    ix2word = dict((i, c) for i, c in enumerate(words))

    # 数据转为index编码
    data_encoded = []
    for name in data:
        name_txt = [word2ix[i] for i in name]
        data_encoded.append(name_txt)
    data_encoded = np.array(data_encoded)

    np.savez('../data/vocabulary.npz', data=data_encoded, word2ix=word2ix, ix2word=ix2word)
    return data_encoded, word2ix, ix2word


def train(model, train_loader, loss_f, optimizer, config):
    print("--- Start training ---")
    model.train()
    for epoch in range(config.epochs):
        avg_loss = 0
        length = len(train_loader)
        start_time = time.time()  # 记录每个epoch的起始时间，用来算耗时
        for batch_data in train_loader:
            batch_data = batch_data.long()
            input_data, target = Variable(batch_data[:, :-1]), Variable(batch_data[:, 1:])  # 行方向错开一位来训练
            pred, _ = model(input_data)
            loss = loss_f(pred, target.contiguous().view(-1))
            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item() / length
            optimizer.step()

        epoch_time = time.time() - start_time
        if (epoch + 1) % config.verbose == 0:
            print(f"Epoch {epoch + 1}/{config.epochs} \t Time used={epoch_time:.1f}s \t Loss={avg_loss:.3f}")
    torch.save(model, '../model.pth')
    print("--- Model saved ---")


if __name__ == "__main__":
    house_data = pd.read_csv('../data/data_cleaned.csv')
    house_data = house_data.loc[(house_data['string_len'] > 1) & (house_data['string_len'] < 9), :].copy()

    data, word2ix, ix2word = prepare_data(house_data, 'alias')

    config = Config()
    model = MyModel(len(word2ix), config.embedding_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # Adam优化
    loss_f = nn.CrossEntropyLoss()  # 交叉熵损失函数

    data = torch.from_numpy(data)
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    train(model, data_loader, loss_f, optimizer, config)
