import numpy as np
import pandas as pd

# 设置随机种子以确保结果可重复
np.random.seed(0)
m = 100
# 生成第一类数据 (均值为 [1, 3]，标准差为 1)
class_0_x = np.random.normal(loc=1, scale=1, size=(m, 1))
class_0_y = np.random.normal(loc=3, scale=1, size=(m, 1))
labels_0 = np.zeros((m, 1))


# 生成第二类数据 (均值为 [3, 1]，标准差为 1)
class_1_x = np.random.normal(loc=3, scale=1, size=(m, 1))
class_1_y = np.random.normal(loc=1, scale=1, size=(m, 1))
labels_1 = np.ones((m, 1))

# 合并数据和标签
data_1 = np.hstack((class_0_x, class_0_y, labels_0))
data_2 = np.hstack((class_1_x, class_1_y, labels_1))
data = np.vstack((data_1, data_2))



# 将数据保存到 CSV 文件
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
df.to_csv('gauss.csv', index=False, header=False)