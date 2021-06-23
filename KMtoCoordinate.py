from getAllData import getMin


# 得到经纬度坐标
def toCoord(lis):  # 最小纬度  最小经度  现在x, y位置
    # latitude, longitude = getMin()
    latitude, longitude = 108.829171, 34.16839
    x = lis[0] / 111.1 + latitude
    y = lis[1] / 92 + longitude
    return [x, y]


def getOldPosition():
    minX, minY = 108.829171, 34.16839
    oldPosition = [[108.96106, 34.310579], [108.963118, 34.172198], [108.899634, 34.203569]]  # 已有的换车站经纬度坐标
    oldX, oldY = [], []
    for _ in oldPosition:
        oldX.append((_[0] - minX) * 111.1)
        oldY.append((_[1] - minY) * 92)
    return oldX, oldY
