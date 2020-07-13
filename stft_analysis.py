import os
track_length = 660000
# genres 폴더내의 장르 명들을 리스트(genres)에 저장
genre_path = 'genres'
genres = [dirname for dirname in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, dirname))]
print(genres)

import numpy as np
import librosa
# 장르를 key로, 장르별 audio track을 요소로하는 리스트가 value인 dictionary 형태의 dataset 생성
dataset = {}
for dirname in genres:
    dir_path = os.path.join(genre_path, dirname)
    dataset.setdefault(dirname, [])
    for filepath in os.listdir(dir_path):
        if filepath.endswith('.au'):
            file, sample_rate = librosa.load(os.path.join(dir_path, filepath))
            dataset[dirname].append(file[:track_length])
sample_rate = 22050
sr = sample_rate

# 재생 길이의 최소값 탐색
from math import floor
play_time_list = []
for genre in dataset:
    play_time = 0
    
    for track in dataset[genre]:
        play_time = len(track)/sample_rate
        play_time_list.append(round(play_time,4))
        
print(min(play_time_list)) # 29.932
track_length = floor(min(play_time_list) * sample_rate)
print(track_length)
# 모든 트랙의 길이를 track_length로 일치시켜야 함

from librosa import stft
for i in [61, 76, 78]: # hiphop 장르에서 tempo 차이가 났던 곡들
    hiphop_stft = stft(dataset['hiphop'][i], n_fft=2048, hop_length=512)
    print(hiphop_stft.shape)
# stft 리턴값의 shape이 같으므로 길이 고려할 필요 없어보임

print(np.abs(hiphop_stft))

from librosa import stft
sample_rate = 22050
sr = sample_rate
genre_stft = np.empty([0, 1025, 1293])
for track in dataset['classical'][:3]:
    track_stft = stft(track, n_fft=2048, hop_length=512)
    genre_stft = np.append(genre_stft, np.expand_dims(track_stft, axis=0), axis=0)


print(track_stft.shape) # STFT has 1025 frequency bins and 1293 frames in time
print(genre_stft.shape)

%matplotlib inline
import matplotlib.pyplot as plt
from librosa.display import specshow
plt.figure(figsize=(10, 20))

for i in range(3):
    plt.subplot(3, 1, i+1)
    D = np.abs(genre_stft[i])
    specshow(librosa.amplitude_to_db(D, ref=np.max),
                              y_axis='log', x_axis='time')
    plt.title('Power spectrogram %d' % i)
    plt.colorbar(format='%+2.0f dB')
plt.show()

print(D.shape)

def plot_specshow(genre, step, n_fft, hop_length):
    from librosa import stft
    sample_rate = 22050
    sr = sample_rate
    datalist = [dataset[genre][i] for i in step]
    n = 1
    
    %matplotlib inline
    import matplotlib.pyplot as plt
    from librosa.display import specshow
    plt.figure(figsize=(10, 20))
        
    for track in datalist:
        track_stft = stft(track, n_fft=n_fft, hop_length=hop_length)
        
        plt.subplot(len(step), 1, n)
        D = np.abs(track_stft)
        specshow(librosa.amplitude_to_db(D, ref=np.max),
                                  y_axis='log', x_axis='time')
        plt.title('Power spectrogram %d' % n)
        plt.colorbar(format='%+2.0f dB')
        n += 1
    plt.show()

plot_specshow('hiphop', [10, 40, 60])

plot_specshow('classical', [10, 40, 60], n_fft=2048, hop_length=2048)

from librosa import stft
hip_stft = stft(dataset['hiphop'][40], n_fft=2048, hop_length=1024)
print(hip_stft.shape)

# 장르별로 np.abs(stft)를 계산하여 data로 저장
# 변수 설정
n_fft = 2048 # FFT window size
hop_length = 1024 # number audio of frames between STFT columns
time_range = 645 # frame in time (width)
frequency = int(1 + n_fft/2) # frequency bins (height)


from librosa import stft
import random

stft_data = np.empty([0, time_range, frequency]) # stft 결과값을 저장할 ndarray
label = np.array([], dtype=int)                  # 장르의 인덱스값을 저장할 ndarray

for genre in dataset:
    n_tracks = len(dataset[genre])
    label_component = genres.index(genre) # genre 명에 대한 genres 리스트의 인덱스
    
    # 장르당 100곡이 있으므로 한 장르당 인덱스 array 길이는 100
    partial_label = np.full(n_tracks, label_component, dtype=int)
    label = np.append(label, partial_label)
    
    for track in sample_data:
        # append하기 위해 차원 증가
        track_stft = np.abs(stft(track)).reshape((1, frequency, time_range)) 
        
        # frequency가 cnn의 채널이므로 전치
        stft_data = np.append(stft_data, np.transpose(track_stft, (0, 2, 1)), axis=0)

from librosa import stft
stft_data = np.empty([0, time_range, frequency])
# stft 데이터는 특정 시간(time_range)에 따른 각 주파수의 강도(frequency). 즉 채널수는 frequency

for track in dataset['country'][:5]:
        track_stft = np.abs(stft(track, n_fft=n_fft, hop_length=hop_length)).reshape((1, frequency, time_range))
        print(track_stft.shape)
        stft_data = np.append(stft_data, np.transpose(track_stft, (0, 2, 1)), axis=0)
        # track_stft는 (1, frequency, time_range)이므로 (1, time_range, frequency)으로 전치
print(stft_data) # 한 면은 한 track의 stft 행렬

from librosa import stft
import random
random.seed(10)

stft_data = np.empty([0, time_range, frequency])
label = np.array([], dtype=int)
print([0, time_range, frequency])

for genre in dataset:
    sample_data = random.sample(dataset[genre], 40)
    n_tracks = len(sample_data) ##
    print(genre, end=' / ')
    label_component = genres.index(genre) # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int) # genre
    label = np.append(label, partial_label)
    
    for track in sample_data: ##
        track_stft = np.abs(stft(track, n_fft=n_fft, hop_length=hop_length)).reshape((1, frequency, time_range))
        stft_data = np.append(stft_data, np.transpose(track_stft, (0, 2, 1)), axis=0)


print(label)
print(stft_data[0].shape)

from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(stft_data, label, train_size=0.75,
                                                            random_state=1, stratify=label)

from keras.utils import np_utils
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(stft_data, label, train_size=0.75,
                                                            random_state=1, stratify=label)

print(train_label)

from collections import Counter
print(Counter(train_label))
print(Counter(train_label[:20]))
print(Counter(train_label[20:40]))
print(Counter(train_label[40:60]))

from keras.utils import np_utils
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

learning_rate = 0.001
batch_size = 20
epochs = 5
print(train_X.shape)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv1D(1025, 10, strides=4, input_shape=(time_range, frequency), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(512, 3, strides=2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=learning_rate)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))

model.predict(test_X)
# 한 장르로 찍는 모델 나옴.. batch size를 더 크게하기 위해 stft의 가중치를 평균 내어 데이터 축소

# 장르별 stft의 가중치 평균값 비교
from librosa import stft
import numpy as np
sample_rate = 22050
sr = sample_rate
track_num = 10
genre1 = dataset['hiphop'][track_num]
genre2 = dataset['rock'][track_num]
genre3 = dataset['metal'][track_num]

genre1_stft = np.abs(stft(genre1))
genre2_stft = np.abs(stft(genre2))
genre3_stft = np.abs(stft(genre3))

genre1_avg = np.average(genre1_stft, axis=0, weights=range(1, 1026))
genre2_avg = np.average(genre2_stft, axis=0, weights=range(1, 1026))
genre3_avg = np.average(genre3_stft, axis=0, weights=range(1, 1026))
print(genre1_avg.shape)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.subplot(3,1,1)
plt.plot(genre1_avg)

plt.subplot(3,1,2)
plt.plot(genre2_avg)

plt.subplot(3,1,3)
plt.plot(genre3_avg)
plt.show()

print(tuple(random.sample(list(range(100)), 3)))

# 한 장르의 트랙별 stft의 가중치 평균값 비교(hpss로 음성데이터 분리) -> 장르별 차이 확인
from librosa import stft
from librosa.effects import hpss
import numpy as np
import random

sample_rate = 22050
sr = sample_rate
genre = 'country'

random.seed(30)
track_num1, track_num2, track_num3 = tuple(random.sample(list(range(100)), 3))
track1_harm, track1_perc = hpss(dataset[genre][track_num1])
track2_harm, track2_perc = hpss(dataset[genre][track_num2])
track3_harm, track3_perc = hpss(dataset[genre][track_num3])

harm1_stft = np.abs(stft(track1_harm))
perc1_stft = np.abs(stft(track1_perc))
harm2_stft = np.abs(stft(track2_harm))
perc2_stft = np.abs(stft(track2_perc))
harm3_stft = np.abs(stft(track3_harm))
perc3_stft = np.abs(stft(track3_perc))

harm1_avg = np.average(harm1_stft, axis=0, weights=range(1, 1026))
perc1_avg = np.average(perc1_stft, axis=0, weights=range(1, 1026))
harm2_avg = np.average(harm2_stft, axis=0, weights=range(1, 1026))
perc2_avg = np.average(perc2_stft, axis=0, weights=range(1, 1026))
harm3_avg = np.average(harm3_stft, axis=0, weights=range(1, 1026))
perc3_avg = np.average(perc3_stft, axis=0, weights=range(1, 1026))

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
plt.subplot(6,1,1)
plt.plot(harm1_avg)
plt.subplot(6,1,2)
plt.plot(perc1_avg, color='r')

plt.subplot(6,1,3)
plt.plot(harm2_avg)
plt.subplot(6,1,4)
plt.plot(perc2_avg, color='r')

plt.subplot(6,1,5)
plt.plot(harm3_avg)
plt.subplot(6,1,6)
plt.plot(perc3_avg, color='r')
plt.show()

def stft_avg(genre, n, seed=10):
    if n > 100 or n <= 0:
        print('0~100의 정수를 입력해야됨')
        return
    
    from librosa import stft
    import numpy as np
    import random

    sample_rate = 22050
    sr = sample_rate
    genre = genre

    random.seed(seed)
    track_num_list = random.sample(list(range(100)), n)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))
    
    for i, index in enumerate(track_num_list):
        track = dataset[genre][index]
        track_stft = np.abs(stft(track))
        stft_avg = np.average(track_stft, axis=0, weights=range(1, 1026))
        plt.subplot(n,1,i+1)
        plt.plot(stft_avg)
    plt.show()

stft_avg('metal', 5)

# rawdata를 hpss로 나누고 나눈 데이터의 stft 가중치 평균값으로 변환하여 stft_avg에 저장
# num_samples=20 -> 5~10분 걸림
from librosa import stft
from librosa.effects import hpss
import numpy as np
import random
random.seed(10)
n_fft = 2048
hop_length = 512
num_samples = 40

time_range = genre_avg.shape[0]
stft_avg = np.empty([0, time_range, 2])
label = np.array([], dtype=int)

for genre in dataset:
    sample_data = random.sample(dataset[genre], num_samples)
    n_tracks = len(sample_data) ##
    print(genre, end=' / ')
    label_component = genres.index(genre) # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int) # genre
    label = np.append(label, partial_label)
    
    for track in sample_data:
        track_harm, track_perc = hpss(track)
        harm_stft = np.abs(stft(track_harm, n_fft=n_fft, hop_length=hop_length))
        perc_stft = np.abs(stft(track_perc, n_fft=n_fft, hop_length=hop_length))
        harm_avg = np.average(harm_stft, axis=0, weights=range(1, 1026)) # shape : (1290,)
        perc_avg = np.average(perc_stft, axis=0, weights=range(1, 1026))
        
        track_avg = np.expand_dims(np.transpose(np.append(harm_avg.reshape((1,1290)),
                                                          perc_avg.reshape((1,1290)),
                                                          axis=0),
                                                (1,0)),
                                   axis=0)
        stft_avg = np.append(stft_avg, track_avg, axis=0)

# train, test set 분리 후 label을 one-hot-encoding
from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(stft_avg, label, train_size=0.8,
                                                            random_state=1, stratify=label) # stratify로 장르수 균일화
print(train_label)

from keras.utils import np_utils
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

# 변수 지정
learning_rate = 0.001
batch_size = 20
epochs = 10
print(train_X.shape)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv1D(32, 10, strides=4, input_shape=(time_range, 2), activation='relu', padding='same'))
model.add(Dropdout(0.2))
model.add(Conv1D(64, 3, strides=2, activation='relu', padding='same'))
model.add(Dropdout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=learning_rate)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))
