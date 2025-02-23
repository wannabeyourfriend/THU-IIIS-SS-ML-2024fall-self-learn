from skimage import io # 图像输入输出
from skimage.color import rgb2lab, lab2rgb # 图像通道转换
from sklearn.neighbors import KNeighborsRegressor # KNN 回归器
import os
import numpy as np
import matplotlib.pyplot as plt

def read_style_image(file_name, size=1):
    img = io.imread(file_name)
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    
    plt.ylabel('Y axis')
    plt.show()

    # 将RGB矩阵转换成LAB表示法的矩阵，大小仍然是W*H*3，三维分别是L、A、B
    img = rgb2lab(img)
    # 取出图像的宽度和高度
    w, h = img.shape[:2]

    X = []
    Y = []
    # 枚举全部可能的中心点
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 保存所有窗口
            X.append(img[x - size: x + size + 1, \
                y - size: y + size + 1, 0].flatten())
            # 保存窗口对应的色彩值a和b
            Y.append(img[x, y, 1:])
    return X, Y 

def rebuild(X, Y, img, size=1):
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
    knn.fit(X, Y)
    # 打印内容图像
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    if img.shape[2] == 4:  # 如果图片是RGBA格式，去除Alpha通道
        img = img[:, :, :3]
        
    # 将内容图像转为LAB表示
    img = rgb2lab(img)
    w, h = img.shape[:2]

    # 初始化输出图像对应的矩阵
    photo = np.zeros([w, h, 3])
    # 枚举内容图像的中心点，保存所有窗口
    print('Constructing window...')
    X = []
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 得到中心点对应的窗口
            window = img[x - size: x + size + 1, \
                y - size: y + size + 1, 0].flatten()
            X.append(window)
    X = np.array(X)

    # 用KNN回归器预测颜色
    print('Predicting...')
   
    pred_ab = knn.predict(X).reshape(w - 2 * size, h - 2 * size, -1)
    # 设置输出图像
    photo[:, :, 0] = img[:, :, 0]
    photo[size: w - size, size: h - size, 1:] = pred_ab

    # 由于最外面size层无法构造窗口，简单起见，我们直接把这些像素裁剪掉
    photo = photo[size: w - size, size: h - size, :]
    return photo

def create_dataset(data_dir='data/image/vangogh', num=3):
    X = []
    Y = []
    files = np.sort(os.listdir(data_dir))
    num = min(num, len(files))
    for file in files[:num]:
        print('reading', file)
        X0, Y0 = read_style_image(os.path.join(data_dir, file))
       
    if len(X0) != len(Y0):
        min_len = min(len(X0), len(Y0))
        print('X0 and Y0 have different length, return the minimum length:', min_len)
        X0 = X0[:min_len]
        Y0 = Y0[:min_len]
    X.extend(X0)
    Y.extend(Y0)
    # 若X和Y为空，返回空数组
    if len(X) == 0:
        raise ValueError('No data found')
    # 若X != Y, 以len(X)\len(Y)中的较小值为shape返回
    
    return np.array(X), np.array(Y)

def procession_image():
    # 静态变量 i，记录处理次数
    if not hasattr(procession_image, "i"):
        procession_image.i = 0  # 初始化静态变量

    # 调用数据集创建函数
    X, Y = create_dataset()

    # 读取原始图片
    content = io.imread('data/image/demo/demo.jpg')

    # 使用模型重建图片
    new_photo = rebuild(X, Y, content)
    new_photo = lab2rgb(new_photo)

    # 创建结果保存目录（如果不存在）
    os.makedirs('result', exist_ok=True)

    # 保存处理结果，文件名为序列化的 i
    output_path = f'result/knnregression_{procession_image.i}.png'
    plt.imsave(output_path, new_photo)

    # 显示处理结果
    fig = plt.figure()
    plt.imshow(new_photo)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Processed Image {procession_image.i}')
    plt.show()

    # 自增静态变量 i
    procession_image.i += 1

