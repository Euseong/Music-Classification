# !pip install librosa
import os

track_length = 660000
# genres 폴더내의 장르 명들을 리스트(genres)에 저장
genre_path = 'genres'
genres = [dirname for dirname in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, dirname))]
genres.remove('.ipynb_checkpoints')
print(genres)

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

# 변수 지정 (예상 최적값 : learning rate=, batch_size=25, epocks=100이상)
# (learning_ratem, batch_size, epochs) = (0.001, 50, 100) // elu-sigmoid-elu-softmax -> 최대 0.58
# (learning_ratem, batch_size, epochs) = (0.003, 50, 100) // elu-sigmoid-elu-softplus -> 최대 0.56
# (learning_ratem, batch_size, epochs) = (0.003, 100, 100) // relu-sigmoid-relu-softplus -> 최대 0.612
# (learning_ratem, batch_size, epochs) = (0.003, 100, 200) // relu-sigmoid-relu-softplus -> 최대 0.632
# (learning_ratem, batch_size, epochs) = (0.003, 150, 200) // relu-sigmoid-relu-softplus -> 최대 0.636
# (learning_ratem, batch_size, epochs) = (0.003, 150, 220) // relu-sigmoid-relu-softplus -> 최대 0.648
# (learning_ratem, batch_size, epochs) = (0.003, 150, 250) // relu-sigmoid-relu-softplus -> 최대 0.676
# (learning_ratem, batch_size, epochs) = (0.003, 150, 300) // relu-sigmoid-relu-softplus -> 최대 0.616
time_range = 1290
learning_rate = 0.003
batch_size = 150
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
with open("cqt_data03.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('cqt_model_weights.npy')

# 1D CNN으로 music의 장르 예측
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, AveragePooling1D
from keras.optimizers import Adagrad
from keras.utils import multi_gpu_model
n_bins = train_X.shape[2]
time_range = 1290
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
model.load_weights('cqt_model_weights.npy')
gpu_model = multi_gpu_model(model, gpus=2)
# gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

pred = gpu_model.predict(test_X)
def broad_predict(prediction, testset):
    import numpy as np
    correct = 0
    false = 0
    false_genre = np.empty([0], dtype=int)
    for i in range(len(testset)):
        if np.isin(testset[i], prediction[i].argsort()[-2:]):
            correct += 1
        else:
            false += 1
            false_genre = np.append(false_genre, testset[i])
    return (correct, false, false_genre)
C, f, false_genre = broad_predict(pred, test_label1)
print('borad_predict 예측결과 : True(%d), False(%d)' % (C, f), round(C/len(test_label),2))

from collections import Counter
false_list = []
for false in false_genre:
    false_list.append(genres[false])
print(Counter(false_list))
