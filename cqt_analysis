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
from librosa import cqt
n_bins = 50
rock_cqt = np.abs(cqt(dataset['rock'][10], n_bins=n_bins))
print(rock_cqt, rock_cqt.shape)
# (n_bins=84, time_range=1290)

classical_cqt = np.abs(cqt(dataset['classical'][10], n_bins=n_bins))

from librosa.display import specshow
import matplotlib.pyplot as plt
rock_plot = librosa.amplitude_to_db(rock_cqt)
classical_plot = librosa.amplitude_to_db(classical_cqt)

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
specshow(rock_plot, x_axis='time', y_axis='cqt_note')
plt.title('rock')

plt.subplot(1,2,2)
specshow(classical_plot, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('classical')
plt.show()

np.expand_dims(rock_cqt.T, axis=0).shape

from librosa import cqt
import numpy as np

hop_length = 512
num_samples = 100
bins_per_octave = 20
n_bins=168
time_range=1290
cqt_data = np.empty([0, time_range, n_bins], dtype='float16')
label = np.array([], dtype=int)

# track별로 cqt 데이터 저장
for genre in dataset:
    n_tracks = num_samples
    print(genre, end=' / ')
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)

    for track in dataset[genre]:
        track_cqt = np.abs(cqt(track, hop_length=hop_length, n_bins=n_bins))
        track_cqt = np.expand_dims(track_cqt.T, axis=0) # cqt_data에 append하기위해 전치 및차원 증가
        cqt_data = np.append(cqt_data, track_cqt, axis=0)

# np.save('cqt_data.npy', cqt_data)

# train, test set 분리 후 label을 one-hot-encoding
import numpy as np

num_samples = 100
label = np.array([], dtype=int)

for genre in genres:
    n_tracks = num_samples  ##
    label_component = genres.index(genre)  # genre 명에 대한 genres 리스트의 인덱스
    partial_label = np.full(n_tracks, label_component, dtype=int)  # genre
    label = np.append(label, partial_label)
    
from sklearn.model_selection import train_test_split
import numpy as np
# cqt_data = np.load('cqt_data.npy')
# cqt_data = np.load('cqt101_data.npy')
cqt_data = np.load('cqt168_data.npy')
# cqt_data = np.load('cqt168_data2.npy')
train_X, test_X, train_label, test_label1 = train_test_split(cqt_data, label, train_size=0.75,
                                                            random_state=100, stratify=label)  # stratify로 장르 수 균일화

from keras.utils import np_utils

train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label1)

# 변수 지정
# (learning_ratem, batch_size, epochs) = (0.003, 50, 100) // relu-sigmoid-relu-softmax -> 0.604
# (learning_ratem, batch_size, epochs) = (0.003, 50, 100) // relu-sigmoid-relu-softplus -> 최대 0.624
# (learning_ratem, batch_size, epochs) = (0.003, 50, 100) // elu-sigmoid-relu-softplus -> 최대 0.620
# (learning_ratem, batch_size, epochs) = (0.002, 50, 200) // relu-sigmoid-relu-softplus -> 최대 0.604
# (learning_ratem, batch_size, epochs) = (0.002, 100, 200) // relu-sigmoid-relu-softplus -> 최대 0.624
# (learning_ratem, batch_size, epochs) = (0.002, 150, 250) // relu-sigmoid-relu-softplus -> 최대 0.604
# (learning_ratem, batch_size, epochs) = (0.002, 100, 250) // relu-sigmoid-relu-softplus -> 최대 0.652
# (learning_ratem, batch_size, epochs) = (0.002, 100, 300) // relu-sigmoid-relu-softplus -> 최대 0.648
# dropout은 0.3이 최적
time_range = 1290
learning_rate = 0.002
batch_size = 100
epochs = 250
print(train_X.shape)

# 1D CNN으로 cqt_data 훈련
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
n_bins = train_X.shape[2]
model = Sequential()
model.add(Conv1D(n_bins*2, 20, strides=5, input_shape=(time_range, n_bins), activation='relu', padding='same'))
model.add(AveragePooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.3))
model.add(Conv1D(n_bins*4, 2, strides=1, activation='sigmoid', padding='same'))
model.add(AveragePooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softplus'))
optimizer = Adagrad(lr=learning_rate)
model.summary()
gpu_model = multi_gpu_model(model, gpus=2)
gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# stop = EarlyStopping(monitor='val_loss', mode='min', patience=3) callbacks=[stop], 
gpu_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))

from keras.models import model_from_json
import os
model_json = model.to_json()
with open("cqt_data02.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights('cqt_data02.npy')

pred = gpu_model.predict(test_X)
# np.save('pred1.npy', pred)

pred = gpu_model.predict(test_X)
def broad_predict(prediction, testset):
    correct = 0
    false = 0
    for i in range(len(testset)):
        if np.isin(testset[i], prediction[i].argsort()[-2:]):
            correct += 1
        else:
            false += 1
    return (correct, false)
C, f = broad_predict(pred, test_label1)
print('borad_predict 예측결과 : True(%d), False(%d)' % (C, f), round(C/len(test_label),2))





# n_bins=101 cqt
from librosa import cqt
import numpy as np

hop_length = 512
num_samples = 100

n_bins=101
time_range=1290
cqt_data = np.empty([0, time_range, n_bins], dtype='float16')
label = np.array([], dtype=int)

# track별로 cqt 데이터 저장
for genre in dataset:    
    print(genre, end=' / ')
    for track in dataset[genre]:
        track_cqt = np.abs(cqt(track, hop_length=hop_length, n_bins=n_bins))
        track_cqt = np.expand_dims(track_cqt.T, axis=0) # cqt_data에 append하기위해 전치 및차원 증가
        cqt_data = np.append(cqt_data, track_cqt, axis=0)
# np.save('cqt101_data.npy', cqt_data)

# n_bins=42 cqt
from librosa import cqt
import numpy as np

hop_length = 512
num_samples = 100

n_bins=42
time_range=1290
cqt_data = np.empty([0, time_range, n_bins], dtype='float16')
label = np.array([], dtype=int)

# track별로 cqt 데이터 저장
for genre in dataset:
    for track in dataset[genre]:
        print(genre, end=' / ')
        track_cqt = np.abs(cqt(track, hop_length=hop_length, n_bins=n_bins))
        track_cqt = np.expand_dims(track_cqt.T, axis=0) # cqt_data에 append하기위해 전치 및차원 증가
        cqt_data = np.append(cqt_data, track_cqt, axis=0)
# np.save('cqt42_data.npy', cqt_data)
