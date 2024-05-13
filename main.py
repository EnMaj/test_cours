import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt

#Предварительная обработка данных пока в разработке
dataset = pd.read_csv("estadistical.csv")
dataset.head()

x_dataset = dataset.drop("Receive/ Not receive credit ",axis=1)
y_dataset = dataset["Receive/ Not receive credit "]

cat_mask = x_dataset.dtypes==object
cat_cols = x_dataset.columns[cat_mask].tolist()

le = preprocessing.LabelEncoder()

x_dataset[cat_cols] = x_dataset[cat_cols].apply(lambda col: le.fit_transform(col))

xtrain, xtest, ytrain, ytest = train_test_split(x_dataset, y_dataset, test_size = 0.3, stratify = y_dataset)

x = xtrain.to_numpy()
y = ytrain.to_numpy()

def normalize(X):
    return (X-np.mean(X))/np.std(X)

x_norm = []
for i in range(len(x)):
    x_norm.append(x[i][4])

x_norm = np.array(x_norm)

x_norm = normalize(x_norm)

for i in range(len(x)):
    x[i][4] = x_norm[i]

for i in range(len(y)):
    y[i] -= 1

dataset = []
for i in range(len(x)):
    dataset.append(([x[i]],y[i]))


#Задаю гиперпараметры
INPUT_DIM = 20  # кол-во признаков
OUT_DIM = 2  # количество классов
H_DIM = 50  # количество нейронов в первом слое
ALPHA = 0.0002
NUM_EPOCHS = 500
BATCH_SIZE = 50


#Функция активации
def relu(t):
    return np.maximum(t, 0)

#Функция преобразования в вероятность
def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)

#Функция ошибки
def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j,yj]=1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)

def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc

#Инициализация весов
W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)

loss_arr = []

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):
        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))

        # Backward
        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # update
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)


accuracy = calc_accuracy()
print('Accuracy', accuracy)

plt.plot(loss_arr)
plt.show()