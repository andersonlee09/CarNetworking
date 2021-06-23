# from coordToKM import getResult
import numpy as np
import pandas as pd

names = ['住宿服务', '体育休闲服务', '公司企业', '商务住宅', '生活服务', '购物服务', '风景名胜', '餐饮服务']


def getResult(url, minX, minY):  # 返回以公里计算的数据
    result, resultX, resultY = [], [], []
    df = pd.read_excel(url, header=0)
    ###################################################
    df = df[df['x'] > 108.829]  # 约束条件  ##
    df = df[df['x'] < 109.1115]
    df = df[df['y'] > 34.13295]  # #########
    df = df[df['y'] < 34.420251]
    ####################################################
    for _ in df['x']:
        resultX.append([(_ - minX) * 111.1])  # 经度转化为公里数
    for _ in df['y']:
        resultY.append([(_ - minY) * 92])
    out = np.array((resultX, resultY)).T
    return list(out[0])  # 很多行两列


# 总体数据的可视化
def allData():
    lis = []
    minX, minY = getMin()
    for _ in names:
        lis += getResult(f'data/{_}.xls', minX, minY)
        # print(lis)
    return lis


def matrixTranspose():
    return np.array(allData()).T


def getMin():  # 返回约束条件的最小值
    lisX, lisY = [], []
    for _ in names:
        df = pd.read_excel(f'data/{_}.xls', header=0)
        ###################################################
        df = df[df['x'] > 108.829]  # 约束条件  ##
        df = df[df['x'] < 109.1115]
        df = df[df['y'] > 34.13295]  # #########
        df = df[df['y'] < 34.420251]
        ###################################################
        # print(min(df['x']), min(df['y']))
        lisX.append(min(df['x']))
        lisY.append(min(df['y']))
    print(f'allMINx:{min(lisX)}, allMINy:{min(lisY)}')
    return min(lisX), min(lisY)


if __name__ == '__main__':
    # allData()
    # getMin()
    print(allData())
    # print(len(allData()))
    print(matrixTranspose().shape)
