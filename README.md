# 楼盘名字生成器
从链家爬取全国楼盘信息，使用LSTM实现房地产楼盘名的AI生成

main.py  生成器主程序

./src存放爬虫、数据处理、模型及训练代码  
crawler.py  全国楼盘信息爬虫  
data_preprocess.py  原始数据处理  
model.py  模型和配置定义  
train.py 模型训练  

./data存放所有数据文件  
city_url.json  链家各城市页面网址  
house_data.csv  爬虫获得的原始数据  
data_cleaned.csv  预处理后的楼盘数据  
vocabulary.npz  数据形式的楼盘名和对应字库  

详细说明待补全  
to be continuted
