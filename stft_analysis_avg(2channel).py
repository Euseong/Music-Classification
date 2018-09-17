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

from librosa import stft
import numpy as np
import random
track = dataset['jazz'][0]
track_stft = np.abs(stft(track))
print(track_stft.shape)
track_avg = np.average(track_stft, axis=0, weights=range(1, 1026))
print(track_avg.shape)

# rawdata를 hpss로 나누고 나눈 데이터의 stft 가중치 평균값으로 변환하여 stft_avg에 저장
# num_samples=20 -> 5~10분 걸림
from librosa import stft
from librosa.effects import hpss
import numpy as np
import random

random.seed(10)
n_fft = 2048
hop_length = 512
num_samples = 100

time_range = track_avg.shape[0]
stft_avg = np.empty([0, time_range, 2])
label = np.array([], dtype=int)

track별로 stft 데이터의 가중치 평균을 계산하여 저장
for genre in dataset:
    n_tracks = num_samples
    print(genre, end=' / ')
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)

    for track in dataset[genre]:
        track_harm, track_perc = hpss(track)
        harm_stft = np.abs(stft(track_harm, n_fft=n_fft, hop_length=hop_length))
        perc_stft = np.abs(stft(track_perc, n_fft=n_fft, hop_length=hop_length))
        harm_avg = np.average(harm_stft, axis=0, weights=range(1, 1026))  # shape : (1290,)
        perc_avg = np.average(perc_stft, axis=0, weights=range(1, 1026))
        track_avg = np.expand_dims(np.transpose(np.append(harm_avg.reshape((1, 1290)),
                                                          perc_avg.reshape((1, 1290)),
                                                          axis=0),
                                                (1, 0)),
                                   axis=0)
        stft_avg = np.append(stft_avg, track_avg, axis=0)

# np.save('stft_avg_2channel.npy', stft_avg)

# train, test set 분리 후 label을 one-hot-encoding
from librosa import stft
import random

random.seed(10)
num_samples = 100
label = np.array([], dtype=int)

for genre in genres:
    n_tracks = num_samples  ##
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)
    
from sklearn.model_selection import train_test_split
import numpy as np
stft_avg = np.load('stft_avg_2channel.npy')
train_X, test_X, train_label, test_label = train_test_split(stft_avg, label, train_size=0.75,
                                                            random_state=40, stratify=label)  # stratify로 장르 수 균일화
print(train_label)
print(len(train_label))

from keras.utils import np_utils

train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

# 변수 지정
time_range = 1290
learning_rate = 0.001
batch_size = 50
epochs = 30
print(train_X.shape)

# 1D CNN으로 stft의 가중치 평균 데이터를 훈련
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv1D(32, 20, strides=5, input_shape=(time_range, 2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, strides=1, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
optimizer = Adagrad(lr=learning_rate)
model.summary()
gpu_model = multi_gpu_model(model, gpus=2)
gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
gpu_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[stop], validation_data=(test_X, test_label))

# accuracy 0.4 부근에서 향상되지 않으므로 0.4에 수렴할때 까지만 학습
# model.save_weights('sftf_2channel_model')
# 위의 weights를 불러오려면 저장 당시의 model과 layer가 동일해야 함

################

from librosa import stft

hip_stft = np.abs(stft(dataset['hiphop'][40], n_fft=2048, hop_length=512))
print(hip_stft.shape)
print(hip_stft)

# stft 변수 설정
n_fft = 2048  # FFT window size
hop_length = 512  # number audio of frames between STFT columns
time_range = 1290  # frame in time (width) - hip_stft.shape[1]
frequency =  1025 # frequency bins (height) - hip_stft.shape[0]

from librosa import stft
import random

random.seed(10)
num_samples = 100
stft_data = np.empty([0, time_range, frequency], dtype='float16')
label = np.array([], dtype=int)
print([0, time_range, frequency])

# stft-amplitude 데이터 추출
for genre in dataset:
    n_tracks = num_samples  ##
    print(genre, end=' / ')
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)

    for track in dataset[genre]:  ##
        track_stft = np.abs(stft(track, n_fft=n_fft, hop_length=hop_length)).reshape((1, frequency, time_range))
        stft_data = np.append(stft_data, np.transpose(track_stft, (0, 2, 1)), axis=0)

# np.save('stft_abs.npy', stft_data)
stft_data = np.load('stft_abs.npy')
print(label.shape)
print(stft_data.shape)

# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_stft_data = np.empty([0,1290,1025], dtype='float16')

for i in range(len(stft_data)):
    scaled_track = np.expand_dims(scaler.fit_transform(stft_data[i]), axis=0)
    scaled_stft_data = np.append(scaled_stft_data, scaled_track, axis=0)
    if i % 100 == 0:
        print(genres[i//100], 'end', end='/')

# np.save('scaled_stft_abs.npy', scaled_stft_data)
scaled_stft_data = np.load('scaled_stft_abs.npy')

from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(scaled_stft_data, label, train_size=0.75, random_state=1, stratify=label)

from keras.utils import np_utils
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

# Keras 변수 지정
learning_rate = 0.01
batch_size = 100
epochs = 20

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adagrad, Adam

model = Sequential()
model.add(Conv1D(1025*2, 40, strides=20, input_shape=(time_range, frequency), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(512, 4, strides=2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=learning_rate)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))

# Conv2D

print(np.expand_dims(scaled_stft_data, axis=3).shape)
stft_2d = np.expand_dims(scaled_stft_data, axis=3)

from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(stft_2d, label, train_size=0.75, random_state=1, stratify=label)

from keras.utils import np_utils
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

# Keras 변수 지정
learning_rate = 0.001
batch_size = 100
epochs = 20

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adagrad, Adam

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(20,10), strides=(5,3), input_shape=(time_range,frequency,1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
optimizer = Adagrad(lr=learning_rate)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))
