import pandas as pd
import re


# 从TOP100地产名单上摘取了主要地产商前缀名，可根据实际情况调整
entp_list = [
        '恒大','碧桂园','万科','保利','融创','中海','华润','绿城','龙湖','世茂','武汉城建','北辰','鲁能','首创','福星惠誉',
        '荣盛','阳光城','金茂','金科','雅居乐','正荣','富力','龙光','绿地','德信','北京城建','中顺','招商','九龙仓',
        '远洋','中骏','蓝光','祥生','奥园','新力','中国铁建','中铁建','佳兆业','禹洲','泰禾','明发','首开','平安',
        '合景泰富','合景','建发','城投','置地','中建','中交','中铁','交投','中梁','中冶','国信','建工','中信',
        '融信','融侨','金辉','红星','苏宁','中粮','^新城','旭辉','宝龙','三盛','新希望','金地','绿都','城开'
    ]

# 不属于商业楼盘需要剔除的数据，可根据实际情况调整
remove_list = ['地块','酒店','博览','物流','奥特莱斯','商业街','电商','创意园','产业园','直销','步行街','商贸城','会展中心']

if __name__ == "__main__":
    origin = pd.read_csv('../data/house_data.csv')
    city_list = origin['city'].unique().tolist()

    # 特殊符号去除
    origin = origin[~origin['name'].str.contains('|'.join(remove_list))].copy()
    origin['name'] = origin['name'].apply(lambda x: x.replace('/', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace('·', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace('｜', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace('•', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace('-', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace('▪', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace('&amp;', ''))
    origin['name'] = origin['name'].apply(lambda x: x.replace(' ', ''))

    # 无意义部分去除
    origin['name'] = origin['name'].apply(lambda x: re.sub(r'[a-zA-Z0-9一二三四五六七八九]期', '', x))
    origin['name'] = origin['name'].apply(lambda x: re.sub('[(\.)(（.*）)(\(.*\))(\|)(（.*\))]', '', x))
    origin['name'] = origin['name'].apply(lambda x: re.sub(r'[a-zA-Z0-9东西南北一二三四五六七八九]+区', '', x))
    origin['name'] = origin['name'].apply(lambda x: re.sub('[a-zA-Z0-9东西南北一二三四五六七八九]、', '', x))
    origin['enterprise'] = origin['name'].apply(
        lambda x: re.search('|'.join(entp_list), x).group() if re.search('|'.join(entp_list), x) else '其他')
    origin['alias'] = origin['name'].apply(lambda x: re.sub('|'.join(entp_list), '', x))
    origin['alias'] = origin['alias'].apply(lambda x: re.sub('|'.join(city_list), '', x))

    origin = origin.drop_duplicates(subset='alias')  # 去重一下
    origin['string_len'] = origin['alias'].str.len()  # 名字长度

    origin.to_csv('../data/data_cleaned.csv', index=False, encoding='utf-8-sig')
