import skimage.measure
import cv2
import os
import numpy as np

image_size = 227
#
# img = cv2.imread('./Images/GADF/Dumbbell_Curl/0_ang.png')
# img = cv2.resize(img, None, fx=image_size / img.shape[0], fy=image_size / img.shape[1])
# entropy1 = skimage.measure.shannon_entropy(img)
# print(entropy1)

# IMG_TYPE = ['GADF', 'MTF']
UPPER_GYM_WORKOUT = ['Dumbbell_Curl', 'Dumbbell_Kickback', 'Hammer_Curl', 'Reverse_Curl']
FEATURES = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ang']

# DC => 0, TK => 1, HC => 2, RC => 3

def Data_Preprocessing_img(IMG_TYPE):
	X_dict ={UPPER_GYM_WORKOUT[0]:[], UPPER_GYM_WORKOUT[1]:[], UPPER_GYM_WORKOUT[2]:[], UPPER_GYM_WORKOUT[3]:[]}

	for label in UPPER_GYM_WORKOUT:
		path = "./Images/" + IMG_TYPE + "/" + label + "/"
		X_ch1 = []
		X_ch2 = []
		X_ch3 = []
		X_ch4 = []
		X_ch5 = []
		X_ch6 = []
		X_ch7 = []
		X_ch8 = []
		X_ang = []
		for top, dir, file in os.walk(path):
			for filename in file:
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
		X_dict[label] = Data_Concatenation(X_ang, X_ch1, X_ch2, X_ch3, X_ch4, X_ch5, X_ch6, X_ch7, X_ch8)

	return X_dict

def Data_Preprocessing_img_Iseries(IMG_TYPE):
	X_dict ={UPPER_GYM_WORKOUT[0]:[], UPPER_GYM_WORKOUT[1]:[], UPPER_GYM_WORKOUT[2]:[], UPPER_GYM_WORKOUT[3]:[]}

	for label in UPPER_GYM_WORKOUT:
		path = "./Images/" + IMG_TYPE + "/" + label + "/"
		X = []
		for top, dir, file in os.walk(path):
			for filename in file:
				img = cv2.imread(path + filename)
				img = cv2.resize(img, None, fx=image_size / img.shape[0], fy=image_size / img.shape[1])
				X.append(img / 256)
		X_dict[label] = X

	return X_dict

def Data_Concatenation(X_ang, X_ch1, X_ch2, X_ch3, X_ch4, X_ch5, X_ch6, X_ch7, X_ch8):
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
    return X

if __name__ == '__main__':
	X_dict = Data_Preprocessing_img_Iseries("I_easy")
	print("***[ I_easy ]***", UPPER_GYM_WORKOUT)
	for workout in UPPER_GYM_WORKOUT:
		entropy = []
		sparse = []
		for i in range(len(X_dict[workout])):
			img = X_dict[workout][i]
			entropy.append(float(skimage.measure.shannon_entropy(img)))
			sparse.append(1 - np.count_nonzero(img)/float(np.size(img)))
		entropy = np.array(entropy)
		sparse = np.array(sparse)
		print(np.mean(entropy), np.std(entropy), min(entropy), max(entropy), np.mean(sparse), np.std(sparse), min(sparse), max(sparse))

	print()
	X_dict = Data_Preprocessing_img_Iseries("I_fair")
	print("***[ I_fair ]***", UPPER_GYM_WORKOUT)
	for workout in UPPER_GYM_WORKOUT:
		entropy = []
		sparse = []
		for i in range(len(X_dict[workout])):
			img = X_dict[workout][i]
			entropy.append(float(skimage.measure.shannon_entropy(img)))
			sparse.append(1 - np.count_nonzero(img)/float(np.size(img)))
		entropy = np.array(entropy)
		sparse = np.array(sparse)
		print(np.mean(entropy), np.std(entropy), min(entropy), max(entropy), np.mean(sparse), np.std(sparse), min(sparse), max(sparse))

	print()
	X_dict = Data_Preprocessing_img_Iseries("I_chal")
	print("***[ I_chal ]***", UPPER_GYM_WORKOUT)
	for workout in UPPER_GYM_WORKOUT:
		entropy = []
		sparse = []
		for i in range(len(X_dict[workout])):
			img = X_dict[workout][i]
			entropy.append(float(skimage.measure.shannon_entropy(img)))
			sparse.append(1 - np.count_nonzero(img)/float(np.size(img)))
		entropy = np.array(entropy)
		sparse = np.array(sparse)
		print(np.mean(entropy), np.std(entropy), min(entropy), max(entropy), np.mean(sparse), np.std(sparse), min(sparse), max(sparse))

	print()
	X_dict = Data_Preprocessing_img("GADF")
	print("***[ GADF ]***", UPPER_GYM_WORKOUT)
	for workout in UPPER_GYM_WORKOUT:
		entropy = []
		sparse = []
		for i in range(len(X_dict[workout])):
			img = X_dict[workout][i]
			entropy.append(float(skimage.measure.shannon_entropy(img)))
			sparse.append(1 - np.count_nonzero(img) / float(np.size(img)))
		entropy = np.array(entropy)
		sparse = np.array(sparse)
		print(np.mean(entropy), np.std(entropy), min(entropy), max(entropy), np.mean(sparse), np.std(sparse), min(sparse), max(sparse))

	print()
	X_dict = Data_Preprocessing_img("MTF")
	print("***[ MTF ]***", UPPER_GYM_WORKOUT)
	for workout in UPPER_GYM_WORKOUT:
		entropy = []
		sparse = []
		for i in range(len(X_dict[workout])):
			img = X_dict[workout][i]
			entropy.append(float(skimage.measure.shannon_entropy(img)))
			sparse.append(1 - np.count_nonzero(img) / float(np.size(img)))
		entropy = np.array(entropy)
		sparse = np.array(sparse)
		print(np.mean(entropy), np.std(entropy), min(entropy), max(entropy), np.mean(sparse), np.std(sparse), min(sparse), max(sparse))