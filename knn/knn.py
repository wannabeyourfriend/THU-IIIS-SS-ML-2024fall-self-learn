import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
m = 20
with open("data/mnist.pkl/mnist.pkl", 'rb') as f:
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
train_x, train_y = train_set[0][:len(train_set[0]) // m], train_set[1][:len(train_set[1]) // m]
valid_x, valid_y = valid_set[0][:len(valid_set[0]) // m], valid_set[1][:len(valid_set[1]) // m]
test_x, test_y = test_set[0][:len(test_set[0]) // m], test_set[1][:len(test_set[1]) // m]
np.random.seed(10)
shuffle_index = np.random.permutation(len(train_x))
train_x, train_y = train_x[shuffle_index], train_y[shuffle_index]

def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def get_knn_indices(self, x):
        dis = list(map(lambda a: distance(a, x), self.train_x))
        knn_indices = np.argsort(dis)[:self.k]
        return knn_indices
    
    def getlabel(self, x):
        knn_indices = self.get_knn_indices(x)
        knn_labels = self.train_y[knn_indices]
        label_count = np.zeros(self.label_num)
        for label in knn_labels:
            label_count[label] += 1
        return np.argmax(label_count)
    
    def predict(self, test_x):
        predicted_test_labels = np.zeros(shape=[len(test_x)], dtype=np.int32)
        for i, x in enumerate(test_x):
            predicted_test_labels[i] = self.getlabel(x)
        return predicted_test_labels
    
if __name__ == '__main__':
    acc = []
    for k in range(1, 11):
        knn = KNN(k, 10)
        knn.fit(train_x, train_y)
        predicted_test_labels = knn.predict(test_x)
        accuracy = np.mean(predicted_test_labels == test_y)
        print(f'K的取值为 {k}, 预测准确率为 {accuracy * 100:.1f}%')
        acc.append(accuracy)
# 绘制准确率曲线
plt.plot(range(1, 11), acc)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy of KNN')
plt.show()