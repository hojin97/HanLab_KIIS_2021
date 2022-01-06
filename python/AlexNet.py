import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image_size = 227

# IMG_TYPE = ['GADF', 'MTF']
UPPER_GYM_WORKOUT = ['Dumbbell_Curl', 'Dumbbell_Kickback', 'Hammer_Curl', 'Reverse_Curl']
FEATURES = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ang']

# DC => 0, TK => 1, HC => 2, RC => 3

def Data_Preprocessing_img(IMG_TYPE):
    X_ch1 = []
    X_ch2 = []
    X_ch3 = []
    X_ch4 = []
    X_ch5 = []
    X_ch6 = []
    X_ch7 = []
    X_ch8 = []
    X_ang = []
    y = []

    count = 0
    for label in UPPER_GYM_WORKOUT:
        path = "./Images/"+IMG_TYPE + "/" + label + "/"
        for top, dir, file in os.walk(path):
            for filename in file:
                # 데이터 랜덤은 여기서,,
                img = cv2.imread(path + filename)
                img = cv2.resize(img, None, fx=image_size / img.shape[0], fy=image_size / img.shape[1])

                if "ang" in filename:
                    X_ang.append(img / 256)

                elif 'ch1' in filename:
                    X_ch1.append(img / 256)

                elif 'ch2' in filename:
                    X_ch2.append(img / 256)

                elif 'ch3' in filename:
                    X_ch3.append(img / 256)

                elif 'ch4' in filename:
                    X_ch4.append(img / 256)

                elif 'ch5' in filename:
                    X_ch5.append(img / 256)

                elif 'ch6' in filename:
                    X_ch6.append(img / 256)

                elif 'ch7' in filename:
                    X_ch7.append(img / 256)

                elif 'ch8' in filename:
                    X_ch8.append(img / 256)

                if count % 9 == 0:
                    if label == 'Dumbbell_Curl':
                        y.append(0)
                    elif label == 'Dumbbell_Kickback':
                        y.append(1)
                    elif label == 'Hammer_Curl':
                        y.append(2)
                    elif label == 'Reverse_Curl':
                        y.append(3)

                count = count + 1
    X, y = Data_Concatenation(X_ang, X_ch1, X_ch2, X_ch3, X_ch4, X_ch5, X_ch6, X_ch7, X_ch8, y)
    return X, y

def Data_Concatenation(X_ang, X_ch1, X_ch2, X_ch3, X_ch4, X_ch5, X_ch6, X_ch7, X_ch8, y):
    X = []
    X_ang = np.array(X_ang, dtype=np.float32)
    X_ch1 = np.array(X_ch1, dtype=np.float32)
    X_ch2 = np.array(X_ch2, dtype=np.float32)
    X_ch3 = np.array(X_ch3, dtype=np.float32)
    X_ch4 = np.array(X_ch4, dtype=np.float32)
    X_ch5 = np.array(X_ch5, dtype=np.float32)
    X_ch6 = np.array(X_ch6, dtype=np.float32)
    X_ch7 = np.array(X_ch7, dtype=np.float32)
    X_ch8 = np.array(X_ch8, dtype=np.float32)

    for i in range(len(X_ang)):
        row = np.concatenate((X_ang[i], X_ch1[i], X_ch2[i], X_ch3[i], X_ch4[i], X_ch5[i], X_ch6[i], X_ch7[i], X_ch8[i]), axis=2)
        X.append(row)

    X = np.array(X)
    y = np.array(y)
    return X, y

def Data_Split(X, y):
    D1 = X[0:9*60*1]
    D2 = X[9*60*1:9*60*2]
    D3 = X[9*60*2:9*60*3]
    D4 = X[9*60*3:]

    D1 = list(D1)
    D2 = list(D2)
    D3 = list(D3)
    D4 = list(D4)
    y = list(y)

    x_index1 = int(60*0.8 * 9)
    x_index2 = int(x_index1 + 60*0.1 * 9)

    y_index1 = int(len(y)*0.8)
    y_index2 = int(y_index1 + len(y)*0.1)

    X_train = D1[0:x_index1]
    X_train.extend(D2[0:x_index1])
    X_train.extend(D3[0:x_index1])
    X_train.extend(D4[0:x_index1])
    y_train = y[0:y_index1]

    X_test = D1[x_index1:x_index2]
    X_test.extend(D2[x_index1:x_index2])
    X_test.extend(D3[x_index1:x_index2])
    X_test.extend(D4[x_index1:x_index2])
    y_test = y[y_index1:y_index2]

    X_validation = D1[x_index2:]
    X_validation.extend(D2[x_index2:])
    X_validation.extend(D3[x_index2:])
    X_validation.extend(D4[x_index2:])
    y_validation = y[y_index2:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_validation), np.array(y_validation)

def makeModel(X):
    AlexNet = keras.models.Sequential([
        keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=X[0].shape),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.BatchNormalization(),

        keras.layers.ZeroPadding2D(2),
        keras.layers.Conv2D(256, (5, 5), strides=1, activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.BatchNormalization(),

        keras.layers.ZeroPadding2D(1),
        keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu'),

        keras.layers.ZeroPadding2D(1),
        keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu'),

        keras.layers.ZeroPadding2D(1),
        keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=2),

        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),

        keras.layers.Dense(4, activation='softmax')
    ])
    return AlexNet

def ProcessAlexnet(iter, IMG_TYPE):
    X, y = Data_Preprocessing_img(IMG_TYPE)
    model = makeModel(X)
    # model.summary()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    train_loss_list = []
    test_acc_list = []
    for i in range(iter):
        # train, validate, test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

        model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_validation, y_validation))

        train_loss, train_acc = model.evaluate(X_train, y_train)
        test_loss, test_acc = model.evaluate(X_test, y_test)

        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)

    return train_loss_list, test_acc_list

def Data_Preprocessing_img_I(IMG_TYPE):
    X = []
    y = []

    for label in UPPER_GYM_WORKOUT:
        path = "./Images/" + IMG_TYPE + "/" + label + "/"
        for top, dir, file in os.walk(path):
            for filename in file:
                img = cv2.imread(path + filename)
                img = cv2.resize(img, None, fx=image_size / img.shape[0], fy=image_size / img.shape[1])

                X.append(img / 256)

                if label == 'Dumbbell_Curl':
                    y.append(0)
                elif label == 'Dumbbell_Kickback':
                    y.append(1)
                elif label == 'Hammer_Curl':
                    y.append(2)
                elif label == 'Reverse_Curl':
                    y.append(3)

    X = np.array(X)
    y = np.array(y)
    return X, y

def ProcessAlexnet_I(iter, IMG_TYPE):
    X, y = Data_Preprocessing_img_I(IMG_TYPE)
    model = makeModel(X)
    # model.summary()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    train_loss_list = []
    test_acc_list = []
    for i in range(iter):
        # train, validate, test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

        model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_validation, y_validation))

        train_loss, train_acc = model.evaluate(X_train, y_train)
        test_loss, test_acc = model.evaluate(X_test, y_test)

        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)

    return train_loss_list, test_acc_list

if __name__ == '__main__':
    GADF_train_loss_list, GADF_test_acc_list = ProcessAlexnet(iter=10, IMG_TYPE="GADF")
    MTF_train_loss_list, MTF_test_acc_list = ProcessAlexnet(iter=10, IMG_TYPE="MTF")
    Ieasy_train_loss_list, Ieasy_test_acc_list = ProcessAlexnet_I(iter=10, IMG_TYPE="I_easy")
    Ifair_train_loss_list, Ifair_test_acc_list = ProcessAlexnet_I(iter=10, IMG_TYPE="I_fair")
    Ichal_train_loss_list, Ichal_test_acc_list = ProcessAlexnet_I(iter=10, IMG_TYPE="I_chal")

    print(GADF_train_loss_list)
    print(GADF_test_acc_list)

    print()
    print()

    print(MTF_train_loss_list)
    print(MTF_test_acc_list)

    print()
    print()

    print(Ieasy_train_loss_list)
    print(Ieasy_test_acc_list)


    print()
    print()

    print(Ifair_train_loss_list)
    print(Ifair_test_acc_list)


    print()
    print()

    print(Ichal_train_loss_list)
    print(Ichal_test_acc_list)

