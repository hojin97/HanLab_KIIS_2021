import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from sklearn.utils import check_array
import os
import numpy as np

# , 'Dumbbell_Kickback', 'Hammer_Curl', 'Reverse_Curl'
UPPER_GYM_WOKROUT = ['Dumbbell_Curl', 'Dumbbell_Kickback', 'Hammer_Curl', 'Reverse_Curl']
IMG_TYPE = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ang']
def file_open():
    data_dict = {'Dumbbell_Curl':[], 'Dumbbell_Kickback':[], 'Hammer_Curl':[], 'Reverse_Curl':[]}
    for workout in UPPER_GYM_WOKROUT:
        path = os.path.join(os.getcwd(), 'data', workout)
        f_list = os.listdir(path)
        for file in f_list:
            sensor_value = []
            read_file = open(os.path.join(path, file), 'r')
            for line in read_file.readlines():
                sensor_value.append(line.split())
            data_dict[workout].append(sensor_value)

    for workout in UPPER_GYM_WOKROUT:
        data_dict[workout] = np.array(data_dict[workout], dtype=np.float32)

    return data_dict

def data2img_GADF(data_dict, workout):
    gadf = GramianAngularField(image_size=16, method='difference')
    fig = plt.figure()

    for index in range(len(data_dict[workout])):
        X_gadf = gadf.fit_transform(data_dict[workout][index])

        for i, data in enumerate(X_gadf):
            f_write = open('./Text/GADF/' + workout + '/' + str(index) + '_' + IMG_TYPE[i] + '.txt', 'w')
            for d_line in data:
                for d in d_line:
                    f_write.write(str(d))
                    f_write.write('\t')
                f_write.write('\n')
            f_write.close()

        # for i, x in enumerate(X_gadf):
        #     plt.imshow(x, cmap='rainbow', origin='lower')
        #     plt.axis('off')
        #     plt.savefig('./Images/GADF/' + workout + '/' + str(index) + '_' +IMG_TYPE[i] + '.png', bbox_inches='tight', pad_inches=0)
        #     # plt.show()
        #     plt.clf()
        print(index)

def data2img_MTF(data_dict, workout):
    mtf = MarkovTransitionField(image_size=16)
    fig = plt.figure()

    for index in range(len(data_dict[workout])):
        X_mtf = mtf.fit_transform(data_dict[workout][index])

        for i, data in enumerate(X_mtf):
            f_write = open('./Text/MTF/' + workout + '/' + str(index) + '_' + IMG_TYPE[i] + '.txt', 'w')
            for d_line in data:
                for d in d_line:
                    f_write.write(str(d))
                    f_write.write('\t')
                f_write.write('\n')
            f_write.close()

        # for i, x in enumerate(X_mtf):
        #     plt.imshow(x, cmap='rainbow', origin='lower')
        #     plt.axis('off')
        #     plt.savefig('./Images/MTF/' + workout + '/' + str(index) + '_' +IMG_TYPE[i] + '.png', bbox_inches='tight', pad_inches=0)
        #     plt.show()
        #     plt.clf()
        print(index)

if '__main__' == __name__:
    data_dict = file_open()
    for workout in UPPER_GYM_WOKROUT:
        data2img_GADF(data_dict, workout)
        print(workout + ":GADF, Done")
        data2img_MTF(data_dict, workout)
        print(workout + ":MTF, Done")
        print()
