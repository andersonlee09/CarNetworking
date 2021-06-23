import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_samples, calinski_harabasz_score
from getAllData import allData, matrixTranspose
import numpy as np
from KMtoCoordinate import getOldPosition, toCoord

names = ['住宿服务', '体育休闲服务', '公司企业', '商务住宅', '生活服务', '购物服务', '风景名胜', '餐饮服务']
oldX, oldY = getOldPosition()  # 获取老的换电站点的X,Y
print(oldX, oldY)
TransposeX = matrixTranspose()[0]  # 获取所有点坐标的X,Y
TransposeY = matrixTranspose()[1]
X = allData()
for i in range(len(X)):
    X[i] = list(X[i])
X = np.array(X)
print(X)
# 总体数据的可视化
for _ in names:
    df = pd.read_excel(f'data/{_}.xls', header=0)
    # print(df)
    plt.scatter(df['x'], df['y'], 10)
    # print(df['x'])
plt.show()

# 开始聚类
for _ in names:
    df = pd.read_excel(f'data/{_}.xls', header=0)
    plt.scatter(TransposeX, TransposeY, c='#e17f0e')
plt.scatter(oldX, oldY, c='purple', marker='*', s=100)
plt.show()

# ######  开始聚类的可视化    ############
inertiaList, silhouetteList, calinski_harabaszList = [], [], []  # 分别为损失值\轮廓系数\卡林斯基-哈拉巴斯指数
for n_cluster in range(26, 27):
    cluster = KMeans(n_clusters=n_cluster).fit(X)
    y_pre = cluster.labels_  # 聚好的类别
    centroid = cluster.cluster_centers_  # 质心
    inertia = cluster.inertia_  # 平方和
    # 可视化聚类效果
    fig, ax1 = plt.subplots(1)
    for i in range(n_cluster):
        ax1.scatter(X[y_pre == i, 0], X[y_pre == i, 1],
                    marker='o',  # 形状
                    s=8)  # 大小
    ax1.scatter(centroid[:, 0], centroid[:, 1],
                marker='x',
                s=20,
                c="black")
    # 添加相应的数值
    inertiaList.append(inertia)
    # silhouetteList.append(silhouette_samples(X, y_pre).mean)      # 计算量大
    calinski_harabaszList.append(calinski_harabasz_score(X, y_pre))  # 卡林斯基-哈拉巴斯指数
    plt.show()
# 可视化平方和效果
print(inertiaList)
plt.plot(range(len(inertiaList)), inertiaList)
plt.show()
# plt.plot(range(len(silhouetteList)), silhouetteList)      # 计算量大
# plt.show()
print(silhouetteList)
plt.plot(range(len(calinski_harabaszList)), calinski_harabaszList)
plt.show()
# 西安1经度92KM，一纬度111.1KM
# print(centroid)

# ######## 开始计算 样本  ################
cir = [[13.88891594, 7.38766948],  # 所有聚类点的坐标
       [26.1138468, 10.03486284],
       [15.79816283, 13.41441379],
       [2.31424195, 11.09615952],
       [24.39393293, 18.18441887],
       [5.43757332, 7.71614628],
       [23.05986024, 8.56186258],
       [18.40999717, 3.54408626],
       [17.47977393, 7.68489168],
       [13.56164585, 17.7090879],
       [8.20941516, 10.91838633],
       [15.17610317, 4.85635192],
       [12.11780506, 5.47874247],
       [30.13802478, 18.84005309],
       [12.53276429, 11.54527792],
       [12.30278489, 14.72138401],
       [4.71876205, 2.98510498],
       [12.38594196, 2.77652436],
       [27.02575846, 13.29857864],
       [7.49614507, 17.56489468],
       [8.12290295, 5.45511264],
       [17.99139104, 19.58578691],
       [9.90803995, 8.14190906],
       [28.25293306, 5.2577948],
       [12.22496407, 8.51460274],
       [20.87816111, 12.55974152]]
bellTower = [13.14635190000031, 8.372919999999738]  # 钟楼点的坐标

# charging = [[14.652867900000114, 13.08138799999952], [14.881511699999116, 0.3503359999999418],
#             [7.828439300000412, 3.2364679999999453]]  # 现有的发电站坐标
charging = [[14.652867900000114, 13.08138799999952], [14.881511699999116, 0.3503359999999418],
            [7.828439300000412, 3.2364679999999453], [23.14635190000031, 17.627080000000262]]  # 现有的发电站坐标


# 现在取 所有 待建点
def getDistance(pos1, pos2):  # 得到两点间的直线距离
    return ((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])) ** 0.5


endCount, endCountPosition = [], []  # 一个记录最小距离点的数目， 一个记录点的位置
for _ in range(0, 11):
    for __ in range(0, 11):
        # 先得到点与现有三个发电站的最小距离
        count, ing = 0, []
        for i in range(0, len(charging)):
            ing.append(getDistance([3.14635190000031 + _ * 2, -2.372919999999738 + __ * 2], charging[i]))
        minStance = min(ing)  # 现有最小距离
        for por in cir:
            if getDistance([3.14635190000031 + _ * 2, -2.372919999999738 + __ * 2], por) < minStance:
                count += 1  # 点数目加1
        endCount.append(count)
        endCountPosition.append([3.14635190000031 + _ * 2, -2.372919999999738 + __ * 2])

goodPosition = []  # 存储好的经纬度坐标
goodCount = []  # 存储周围点的数目
temp = []  # 存储实际公里坐标
for i in range(0, len(endCount)):
    if endCount[i] >= 7:
        goodPosition.append(toCoord(endCountPosition[i]))
        temp.append(endCountPosition[i])
        goodCount.append(endCount[i])
print(len(goodPosition))
# temp = np.array([[23.14635190000031, 17.627080000000262]])    # 第一次点的坐标
# temp = np.array([[7.14635190000031, 9.627080000000262]])  # 第二次点的坐标
for n_cluster in range(26, 27):
    cluster = KMeans(n_clusters=n_cluster).fit(X)
    y_pre = cluster.labels_  # 聚好的类别
    fig, ax1 = plt.subplots(1)
    for i in range(n_cluster):
        ax1.scatter(X[y_pre == i, 0], X[y_pre == i, 1],
                    marker='o',  # 形状
                    s=8)  # 大小
    # ax1.scatter(temp[:, 0], temp[:, 1],
    #             marker='*',
    #             s=120,
    #             c="red")
    temp = np.array(temp)
    ax1.scatter(temp[:, 0], temp[:, 1],
                marker='x',
                s=80,
                c="black")
    charging = np.array(charging)
    for i in range(len(charging)):
        ax1.scatter(charging[:, 0], charging[:, 1],
                    marker='*',
                    s=120,
                    c="red")
plt.show()

allCount = []
su = 0
for _ in goodPosition:
    x, y = _[0], _[1]
    p = 999999999999999
    if x < 108.922566 and y < 34.2904:
        p = 1550
    elif 108.922566 < x < 109.013 and 34.349819 < y:
        p = 2000
    elif 109.013 < x < 109.080775 and 34.349819 < y:
        p = 1550
    elif 108.922566 < x < 108.973 and 34.2163 < y < 34.349819:
        p = 3350
    elif x < 108.922566 and 34.2163 < y < 34.2904:
        p = 2500
    elif 108.973 < x < 109.040055 and 34.2163 < y < 34.349819:
        p = 2500
    elif 109.040055 < x < 109.080775 and 34.19505 < y < 34.349819:
        p = 2000
    elif x < 108.865 and 34.192708 < y < 34.2163:
        p = 2000
    elif 108.865 < x < 108.8889 and 34.192708 < y < 34.2163:
        p = 2500
    elif 108.8889 < x < 108.939 and 34.192708 < y < 34.2163:
        p = 3350
    elif 108.939 < x < 108.954 and 34.192708 < y < 34.2163:
        p = 4500
    elif 108.954 < x < 108.9985 and 34.192708 < y < 34.2163:
        p = 3350
    elif 108.9985 < x < 109.040055 and 34.19505 < y < 34.2163:
        p = 2500
    elif x < 108.865 and y < 34.192708:
        p = 1550
    elif 108.865 < x < 108.939 and 34.19505 < y < 34.192708:
        p = 2500
    elif 108.939 < x < 108.9985 and 34.19505 < y < 34.192708:
        p = 3350
    elif 108.865 < x < 109.040055 and y < 34.19505:
        p = 2500
    elif 109.040055 < x and y < 34.19505:
        p = 1500
    cost1 = 53 * p
    cost2 = 150 * goodCount[su]
    cost3 = 5000 * 2 + 1000000
    # print(cost1 + cost2 + cost3)
    allCount.append(cost1 + cost2 + cost3)
    su += 1
