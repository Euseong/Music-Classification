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

from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff, zero_crossing_rate
from librosa.effects import hpss
import numpy as np
jazz_h, jazz_p = hpss(dataset['jazz'][0])
shape1 = spectral_centroid(jazz_h).shape
shape2 = spectral_bandwidth(jazz_h).shape
shape3 = spectral_flatness(jazz_h).shape
shape4 = spectral_rolloff(jazz_h).shape
shape5 = zero_crossing_rate(jazz_h).shape
print(shape1, shape2, shape3, shape4, shape5)

time_range = 1290
centroid = spectral_centroid(jazz_h)
bandwidth = spectral_bandwidth(jazz_h)
flatness = spectral_flatness(jazz_h)
rolloff = spectral_rolloff(jazz_h)
zero = zero_crossing_rate(jazz_h)

print(centroid)
print(bandwidth)
print(flatness)
print(rolloff)
track_feature = np.array([centroid, bandwidth, flatness, rolloff, zero], dtype='float16')
track_feature = np.squeeze(track_feature, axis=1)
print(np.log(track_feature.T))

a = np.array([3,2,0, 5])
np.log(a, where=(a != 0))


# rawdata를 hpss로 나누고 나눈 데이터의 spectral feature들을 계산
from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff, zero_crossing_rate
from librosa.effects import hpss
import numpy as np

n_fft = 2048
hop_length = 512
num_samples = 100
channel = 5
time_range = 1290
spectral_features = np.empty([0, time_range, channel])
label = np.array([], dtype=int)

# track별로 stft 데이터의 가중치 평균을 계산하여 저장
for genre in dataset:
    n_tracks = num_samples
    print(genre, end=' / ')
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)

    for track in dataset[genre]:
        centroid = spectral_centroid(track)
        bandwidth = spectral_bandwidth(track)
        flatness = spectral_flatness(track)
        rolloff = spectral_rolloff(track)
        zero = zero_crossing_rate(track)
        track_feature = np.array([centroid, bandwidth, flatness, rolloff, zero], dtype='float16')
        track_feature = np.expand_dims(np.squeeze(track_feature, axis=1).T, axis=0)
        spectral_features = np.append(spectral_features, track_feature, axis=0)
        

print(spectral_features)

# np.save('spectral_features.npy', spectral_features)
# spectral_features_log = np.log(spectral_features, where=(spectral_features != 0))
# np.save('spectral_features_log.npy', spectral_features_log)

spectral_features = np.load('spectral_features.npy')
mfcc_data = np.load('mfcc_data.npy')
print(spectral_features[0,0])
print(mfcc_data[0,0])
print(np.append(spectral_features, mfcc_data, axis=2)[0,0])

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
spectral_features = np.load('spectral_features.npy')
mfcc_data = np.load('mfcc_data.npy')
data = np.append(spectral_features, mfcc_data, axis=2)
# spectral_features = np.load('spectral_features_log.npy')
train_X, test_X, train_label1, test_label1 = train_test_split(data, label, train_size=0.75, random_state=100, stratify=label)
print(len(train_label1))

from keras.utils import np_utils

train_label = np_utils.to_categorical(train_label1)
test_label = np_utils.to_categorical(test_label1)

# 변수 지정
print(time_range)
learning_rate = 0.005
batch_size = 50
epochs = 50
print(train_X.shape)

#### 1D CNN으로 mfcc_data 훈련
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.optimizers import RMSprop, Adagrad, Adam, SGD, Adadelta, Adamax
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
# mfcc 값이 음수값도 다수 포함하기 때문에 tanh로 학습
model = Sequential()
model.add(Conv1D(10*3, 20, strides=5, input_shape=(time_range, 10), activation='tanh', padding='same'))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(10*3*2, 2, strides=1, activation='tanh', padding='same'))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(50, activation='tanh'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
optimizer = Adagrad(lr=learning_rate)
model.summary()
gpu_model = multi_gpu_model(model, gpus=2)
gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# stop = EarlyStopping(monitor='val_loss', mode='min', patience=3) callbacks=[stop], 
gpu_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))

pred2 = gpu_model.predict(test_X)

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
