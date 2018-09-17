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
stft_avg = np.empty([0, time_range, 1])
label = np.array([], dtype=int)


for genre in dataset:
    n_tracks = num_samples
    print(genre, end=' / ')
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)

    for track in dataset[genre]:
        track_harm, track_perc = hpss(track)
        perc_stft = np.abs(stft(track_perc, n_fft=n_fft, hop_length=hop_length))
        perc_avg = np.average(perc_stft, axis=0, weights=range(1, 1026))
        track_avg = np.expand_dims(perc_avg.reshape((1, 1290)).T, axis=0)
        stft_avg = np.append(stft_avg, track_avg, axis=0)

# np.save('sftf_svg_1channel.npy', stft_avg)

# train, test set 분리 후 label을 one-hot-encoding
from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(stft_avg, label, train_size=0.75,
                                                            random_state=1, stratify=label)  # stratify로 장르 수 균일화
print(train_label)
print(len(train_label))

from keras.utils import np_utils

train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

# 변수 지정
learning_rate = 0.005
batch_size = 30
epochs = 100
print(train_X.shape)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.utils import multi_gpu_model

model = Sequential()
model.add(Conv1D(64, 100, strides=50, input_shape=(time_range, 1), activation='relu', padding='same'))
model.add(Conv1D(128, 3, strides=2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
optimizer = Adagrad(lr=learning_rate)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(test_X, test_label))



from librosa import stft

hip_stft = stft(dataset['hiphop'][40], n_fft=2048, hop_length=1024)
print(hip_stft.shape)
