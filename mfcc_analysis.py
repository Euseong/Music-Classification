# !pip install librosa
import os

track_length = 660000
# genres 폴더내의 장르 명들을 리스트(genres)에 저장
genre_path = 'genres'
genres = [dirname for dirname in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, dirname))]
genres.remove('.ipynb_checkpoints')
print(genres)

import numpy as np
import librosa

# 장르를 key로, 장르별 audio track을 요소로하는 리스트가 value인 dictionary 형태의 dataset 생성
dataset = {}
for dirname in genres:
    dir_path = os.path.join(genre_path, dirname)
    dataset.setdefault(dirname, [])
    for filepath in os.listdir(dir_path):
        if filepath.endswith('.au'):  # 음악파일의 확장자와 일치하는 경우만 dataset에 추가
            file, sample_rate = librosa.load(os.path.join(dir_path, filepath))
            dataset[dirname].append(file[:track_length])
sample_rate = 22050
sr = sample_rate



# cqt는 freqeuncy resolution을 줄여줌
from librosa.feature import mfcc
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,100))

n_mfcc = 20
rock_mfcc = scaler.fit_transform(mfcc(dataset['rock'][10], n_mfcc=n_mfcc))
print(rock_mfcc, rock_mfcc.shape)
# (n_mfcc=20, time_range=1290)

classical_mfcc = scaler.fit_transform(mfcc(dataset['classical'][10], n_mfcc=n_mfcc))
from librosa.display import specshow
import matplotlib.pyplot as plt

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
specshow(rock_mfcc, x_axis='time')
plt.title('rock')

plt.subplot(1,2,2)
specshow(classical_mfcc, x_axis='time')
plt.title('classical')
plt.colorbar()
plt.show()

plt.figure(figsize=(15,20))
n = 1
for genre in dataset:
    genre_mfcc = scaler.fit_transform(mfcc(dataset[genre][10], n_mfcc=20))
    plt.subplot(5, 2, n)
    specshow(genre_mfcc, x_axis='time')
    plt.title('%s MFCC' % genre)
    plt.colorbar()
    n += 1
plt.show()

plt.figure(figsize=(15,20))
n = 1
for i in range(10):
    jazz_mfcc = scaler.fit_transform(mfcc(dataset['jazz'][i+10], n_mfcc=5))
    plt.subplot(5, 2, n)
    specshow(jazz_mfcc, x_axis='time')
    plt.title('jazz MFCC')
    plt.colorbar()
    n += 1
plt.show()

# MFCC 그래프의 아래 5개 정도 그룹이 장르별로 특징을 가짐

from librosa.feature import mfcc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_samples = 100
hop_length = 512
n_mfcc = 5
time_range=1290
mfcc_data = np.empty([0, time_range, n_mfcc], dtype='float16')
label = np.array([], dtype=int)

# track별로 mfcc 데이터 저장
for genre in dataset:
    n_tracks = num_samples
    print(genre, end=' / ')
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)

    for track in dataset[genre]:
        track_mfcc = scaler.fit_transform(mfcc(track, hop_length=hop_length, n_mfcc=n_mfcc))
        track_mfcc = np.expand_dims(track_mfcc.T, axis=0) # mfcc_data에 append하기위해 전치 및차원 증가
        mfcc_data = np.append(mfcc_data, track_mfcc, axis=0)

# np.save('mfcc_data.npy', mfcc_data)
# np.save('mfcc5_data_scaled.npy', mfcc_data)
# np.save('mfcc10_data_scaled.npy', mfcc_data)

print(mfcc_data)

# train, test set 분리 후 label을 one-hot-encoding
# import numpy as np
# num_samples = 100
# label = np.array([], dtype=int)

# for genre in genres:
#     n_tracks = num_samples  ##
#     label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
#     partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
#     label = np.append(label, partial_label)
    
from sklearn.model_selection import train_test_split
import numpy as np
# mfcc_data = np.load('mfcc_data_scaled.npy')
train_X, test_X, train_label1, test_label1 = train_test_split(mfcc_data, label, train_size=0.75, random_state=100, stratify=label)
print(train_label)
print(len(train_label))

from keras.utils import np_utils

train_label = np_utils.to_categorical(train_label1)
test_label = np_utils.to_categorical(test_label1)

# 변수 지정
time_range = 1290
learning_rate = 0.01
batch_size = 25
epochs = 40
print(train_X.shape)

#### 1D CNN으로 mfcc_data 훈련
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
# mfcc 값이 음수값도 다수 포함하기 때문에 tanh로 학습
model = Sequential()
model.add(Conv1D(5*3, 20, strides=5, input_shape=(time_range, 5), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(5*3*2, 2, strides=1, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='softmax'))
optimizer = Adagrad(lr=learning_rate)
model.summary()
gpu_model = multi_gpu_model(model, gpus=2)
gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# stop = EarlyStopping(monitor='val_loss', mode='min', patience=3) callbacks=[stop], 
gpu_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))

from keras.models import model_from_json
import os
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights('mfcc_model')

pred2 = gpu_model.predict(test_X)
print(np.argmax(pred2, axis=1))
print(np.equal(test_label1, np.argmax(pred2, axis=1)))

n = 30
print('',test_label1[:n])
print(pred1[:n].argsort(axis=1)[:, -2:].T)
print(pred2[:n].argsort(axis=1)[:, -2:].T)
print(sum_pred[:n].argsort(axis=1)[:, -2:].T)

pred1 = np.load('pred1.npy')
print(np.round(pred1[0],2))
print(np.round(pred2[0],2))
sum_pred = np.add(pred1, pred2)
print(np.round(sum_pred, 2)[0])

def broad_predict(prediction, testset):
    correct = 0
    false = 0
    for i in range(len(testset)):
        if np.isin(testset[i], prediction[i].argsort()[-2:]):
            correct += 1
        else:
            false += 1
    return (correct, false)

C1, f1 = broad_predict(pred1, test_label1)
C2, f2 = broad_predict(pred2, test_label1)
C3, f3 = broad_predict(sum_pred, test_label1)
print('pred1 예측결과 : True(%d), False(%d)' % (C1, f1), round(C1/len(test_label1),2))
print('pred2 예측결과 : True(%d), False(%d)' % (C2, f2), round(C2/len(test_label1),2))
print('sum_pred 예측결과 : True(%d), False(%d)' % (C3, f3), round(C3/len(test_label1),2))

False == np.isin(5,pred1.argsort(axis=1)[3, -2:])
