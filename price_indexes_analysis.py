# /usr/bin/python3
# coding: utf-8

from operator import le
import os
import sys
from tkinter import N
import numpy as np
from sklearn import cluster
from pypinyin import pinyin
import sklearn

def read_house_price(txt_path, name_py = True):
    # 读取每个城市的房价环比涨幅
    prices = {}
    with open(txt_path) as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.replace("*", "").strip().strip(" ").replace(" ", "")
            data = [d for d in line.split("	") if len(d) > 0]
            if len(data) != 8:
                continue
            city1, huanbi1, tongbi1, dingji1, city2, huanbi2, tongbi2, dingji2 = data 
            city1 = city1.replace(" ", "").replace("　", "")
            city2 = city2.replace(" ", "").replace("　", "")
            # 北　　京	100.3 	102.6 	101.2 	唐　　山	102.1 	103.0	102.3 
            if name_py:
                city2 = "".join([p[0] for p in pinyin(city2)])
                city1 = "".join([p[0] for p in pinyin(city1)])
            if city1 not in prices:
                prices[city1] = []
            if city2 not in prices:
                prices[city2] = []
            prices[city1].append(float(huanbi1))
            prices[city2].append(float(huanbi2))

    cur_prices = {}
    for key, value in prices.items():
        prices[key] = np.array(value)
        cur_price = [1.0]
        for v in value:
            cur_price.append(cur_price[-1] * v/100.0)
        cur_prices[key] = np.array(cur_price[1:])

    return prices, cur_prices


def read_city_levels(city_level_txt, cur_prices):
    # 读取城市等级
    city_levels = {}
    level = None
    with open(city_level_txt) as fr:
        for line in fr.readlines():
            if "线城市" in line:
                level = line.strip()
            else:
                if len(line.strip()) > 0:
                    city_name = "".join([p[0] for p in pinyin(line.strip())])
                    city_levels[city_name] = level

    # 计算不同等级的城市的房价曲线
    cur_prices_levels = {}
    for city_name, prices in cur_prices.items():
        city_found = False
        for key, level in city_levels.items():
            if city_name in key:
                city_found = True
                level_name = "".join([p[0] for p in pinyin(level.strip()[:-2])])
                if level_name not in cur_prices_levels:
                    cur_prices_levels[level_name] = []
                print(city_name, level_name)
                cur_prices_levels[level_name].append(prices)
        if not city_found:
            print("city not found", city_name)
    for level_name, prices in cur_prices_levels.items():
        print("=========", level_name, len(prices))
        cur_prices_levels[level_name] = np.mean(prices, axis=0)
    return cur_prices_levels

def read_city_clusters(cur_prices, prices_changes = None):
    estimator = sklearn.cluster.KMeans(n_clusters=4,
	 init='k-means++', 
	n_init=100, 
	max_iter=1000, 
	tol=0.0001, 
	precompute_distances='auto', 
	verbose=0, 
	random_state=None, 
	copy_x=True, 
	n_jobs=1, 
	algorithm='auto'
	)

    data = []
    city_names = []
    for city_name, prices in cur_prices.items():
        if prices_changes is not None:
            data.append(prices_changes[city_name])
        else:
            data.append(prices)
        city_names.append(city_name)
    data = np.array(data)
    # data_norm = (data - np.mean(data, axis=1, keepdims=True))/np.std(data, axis=1, keepdims=True)
    estimator.fit(data)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和

    cluster_prices = {}
    cluster_names = {}
    for i in range(len(city_names)):
        prices = cur_prices[city_names[i]]
        label = label_pred[i]
        cluster_name = "cluster_{}".format(label)
        if cluster_name not in cluster_prices:
            cluster_prices[cluster_name] = []
            cluster_names[cluster_name] = []
        cluster_prices[cluster_name].append(prices)
        cluster_names[cluster_name].append(city_names[i])
    for key, value in cluster_prices.items():
        cluster_prices[key] = np.mean(value, axis=0)
        print(key, cluster_names[key])
    return cluster_prices, cluster_names

def read_house_area_price(txt_path, city = "南京", name_py=False):
    price_changes = {"<90":[], "90-144":[], ">144":[]}
    with open(txt_path) as fr:
        lines = fr.readlines()
        for line in lines:
            if city[0] in line and city[1] in line:
                # 南　　京        99.4    103.6   105.3   99.5    103.3   105.2   99.6    105.5   107.0
                line = line.strip().replace(" ", "")
                data = [d for d in line.split("	") if len(d.strip()) > 0]
                assert len(data) == 10
                price_changes["<90"].append(float(data[1]) if data[1] != "--" else 100.0)
                price_changes["90-144"].append(float(data[4]) if data[4] != "--" else 100.0)
                price_changes[">144"].append(float(data[7]) if data[7] != "--" else 100.0)

    cur_prices = {}
    for key, value in price_changes.items():
        price_changes[key] = np.array(value)
        cur_price = [1.0]
        for v in value:
            cur_price.append(cur_price[-1] * v/100.0)
        cur_prices[key] = np.array(cur_price[1:])

    return price_changes, cur_prices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

# # =================== 新房价格曲线 =======================
# root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# prices, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市新建商品住宅销售价格指数.txt"))
# dates = []
# for year in range(2011, 2023):
#     for month in range(1, 13):
#         if year == 2022 and month > 4:
#             continue
#         dates.append("{:04d}{:02d}".format(year, month))

# x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# days = mdates.DayLocator()

# print("start plot")
# # 绘制所有城市的的新房价格
# fig, ax = plt.subplots()   
# legends = []
# color_map = plt.cm.get_cmap('hsv', 100)
# cur_prices_list = [(key, value)for key, value in cur_prices.items()]
# cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
# for i, (city, prices) in enumerate(cur_prices_list):
#     print(i, city, prices[-1])
#     ax.plot(x_values, prices, c = color_map(i))
#     legends.append(city)
# ax.legend(legends, loc = 2, ncol=2, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlabel("Date")
# ax.set_ylabel('Prices')
# plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
# plt.show()



# # =================== 全国二手房房价格曲线 =======================
# root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# prices, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市二手住宅销售价格指数.txt"))


# dates = []
# for year in range(2011, 2023):
#     for month in range(1, 13):
#         if year == 2022 and month > 4:
#             continue
#         dates.append("{:04d}{:02d}".format(year, month))

# x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# days = mdates.DayLocator()

# print("start plot")
# # 绘制所有城市的的新房价格
# fig, ax = plt.subplots()   
# legends = []
# color_map = plt.cm.get_cmap('hsv', 100)
# cur_prices_list = [(key, value)for key, value in cur_prices.items()]
# cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
# for i, (city, prices) in enumerate(cur_prices_list):
#     print(i, city, prices[-1])
#     ax.plot(x_values, prices, c = color_map(i))
#     legends.append(city)
# ax.legend(legends, loc = 2, ncol=2, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlabel("Date")
# ax.set_ylabel('Prices')
# plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
# plt.show()





# =================== 一线、新一线、二线新房房价格曲线 =======================
# root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# prices, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市新建商品住宅销售价格指数.txt"))
# cur_prices_five_levels = read_city_levels(os.path.join(root_dir, "城市分级名单.txt"), cur_prices)

# dates = []
# for year in range(2011, 2023):
#     for month in range(1, 13):
#         if year == 2022 and month > 4:
#             continue
#         dates.append("{:04d}{:02d}".format(year, month))

# x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# days = mdates.DayLocator()

# print("start plot")
# # 绘制所有城市的的新房价格
# fig, ax = plt.subplots()   
# legends = []
# color_map = plt.cm.get_cmap('hsv', 10)
# cur_prices_list = [(key, value)for key, value in cur_prices_five_levels.items()]
# cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
# for i, (city, prices) in enumerate(cur_prices_list):
#     print(i, city, prices[-1])
#     ax.plot(x_values, prices, c = color_map(i))
#     legends.append(city)
# ax.legend(legends, loc = 2, ncol=1, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlabel("Date")
# ax.set_ylabel('Prices')
# plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
# plt.show()




# =================== 一线、新一线、二线二手房房价格曲线 =======================
# root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# prices, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市二手住宅销售价格指数.txt"))
# cur_prices_five_levels = read_city_levels(os.path.join(root_dir, "城市分级名单.txt"), cur_prices)

# dates = []
# for year in range(2011, 2023):
#     for month in range(1, 13):
#         if year == 2022 and month > 4:
#             continue
#         dates.append("{:04d}{:02d}".format(year, month))

# x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# days = mdates.DayLocator()

# print("start plot")
# fig, ax = plt.subplots()   
# legends = []
# color_map = plt.cm.get_cmap('hsv', 10)
# cur_prices_list = [(key, value)for key, value in cur_prices_five_levels.items()]
# cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
# for i, (city, prices) in enumerate(cur_prices_list):
#     print(i, city, prices[-1], np.max(prices))
#     ax.plot(x_values, prices, c = color_map(i))
#     legends.append(city)
# ax.legend(legends, loc = 2, ncol=1, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlabel("Date")
# ax.set_ylabel('Prices')
# plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
# plt.show()




# # =================== 一线、新一线、二线新房房价格聚类 =======================
# root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# prices_changes, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市新建商品住宅销售价格指数.txt"), name_py=False)
# # _, cur_prices_ershou = read_house_price(os.path.join(root_dir, "70个大中城市二手住宅销售价格指数.txt"), name_py=False)
# cur_prices_four_clusters, cluster_names = read_city_clusters(cur_prices, prices_changes) # , cur_prices_ershou)

# dates = []
# for year in range(2011, 2023):
#     for month in range(1, 13):
#         if year == 2022 and month > 4:
#             continue
#         dates.append("{:04d}{:02d}".format(year, month))

# x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# days = mdates.DayLocator()

# print("start plot")
# # 绘制所有城市的的新房价格
# fig, ax = plt.subplots()   
# legends = []
# color_map = plt.cm.get_cmap('hsv', 5)
# cur_prices_list = [(key, value)for key, value in cur_prices_four_clusters.items()]
# cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
# for i, (city, prices) in enumerate(cur_prices_list):
#     print(i, city, prices[-1])
#     ax.plot(x_values, prices, c = color_map(i))
#     legends.append(city)
# ax.legend(legends, loc = 2, ncol=1, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlabel("Date")
# ax.set_ylabel('Prices')
# plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
# plt.show()





# =================== 一线、新一线、二线新房房价格聚类 =======================
# root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# # prices_changes, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市新建商品住宅销售价格指数.txt"), name_py=False)
# prices_changes, cur_prices = read_house_price(os.path.join(root_dir, "70个大中城市二手住宅销售价格指数.txt"), name_py=False)
# cur_prices_four_clusters, cluster_names = read_city_clusters(cur_prices, prices_changes) # , cur_prices_ershou)

# dates = []
# for year in range(2011, 2023):
#     for month in range(1, 13):
#         if year == 2022 and month > 4:
#             continue
#         dates.append("{:04d}{:02d}".format(year, month))

# x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# days = mdates.DayLocator()

# print("start plot")
# # 绘制所有城市的的新房价格
# fig, ax = plt.subplots()   
# legends = []
# color_map = plt.cm.get_cmap('hsv', 5)
# cur_prices_list = [(key, value)for key, value in cur_prices_four_clusters.items()]
# cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
# for i, (city, prices) in enumerate(cur_prices_list):
#     print(i, city, prices[-1])
#     ax.plot(x_values, prices, c = color_map(i))
#     legends.append(city)
# ax.legend(legends, loc = 2, ncol=1, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlabel("Date")
# ax.set_ylabel('Prices')
# plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
# plt.show()



# 南京不同面积的房子价格变化分析
root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# prices_changes, cur_prices = read_house_area_price(os.path.join(root_dir, "70个大中城市新建商品住宅销售价格分类指数.txt"), city = "南京", name_py=False)
prices_changes, cur_prices = read_house_area_price(os.path.join(root_dir, "70个大中城市二手住宅销售价格分类指数.txt"), city = "南京", name_py=False)
dates = []
for year in range(2011, 2023):
    for month in range(1, 13):
        if year == 2022 and month > 4:
            continue
        dates.append("{:04d}{:02d}".format(year, month))

x_values = [datetime.strptime(str(d), '%Y%m').date() for d in dates]
years = mdates.YearLocator()
months = mdates.MonthLocator()
days = mdates.DayLocator()

print("start plot")
# 绘制所有城市的的新房价格
fig, ax = plt.subplots()   
legends = []
color_map = plt.cm.get_cmap('hsv', 5)
cur_prices_list = [(key, value)for key, value in cur_prices.items()]
cur_prices_list = sorted(cur_prices_list, key=lambda s: s[1][-1], reverse=True)  
for i, (city, prices) in enumerate(cur_prices_list):
    print(i, city, prices[-1], np.max(prices))
    ax.plot(x_values, prices, c = color_map(i))
    legends.append(city)
ax.legend(legends, loc = 2, ncol=1, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15)) # interval = 1
ax.xaxis.set_major_locator(years)
ax.xaxis.set_minor_locator(months)
ax.set_xlabel("Date")
ax.set_ylabel('Prices')
plt.xticks(rotation=-20)    # 设置x轴标签旋转角度
plt.show()
