# !pip install pydub

import librosa
from pydub import AudioSegment
import numpy as np
# download from - http://freemusicarchive.org/
music = 'music_samples/blues/blues1.mp3' # 장르를 예측할 음악 파일 경로 입력
sample_rate = AudioSegment.from_mp3(music).frame_rate
time_segment = 660000 # 훈련 데이터의 time_segment
print(sample_rate)
music_data, sample_rate = librosa.load(music, sr=sample_rate)
print(len(music_data))
# music_segment

a = np.array([1,2,3,4,5,6,7,8,9,0])
np.array_split(a, 2)

print(len(music_data)/time_segment)
# music_data의 총 길이가 time_segment로 나누어 떨어지지 않으면 segment의 개수는 그 몫 + 1
num_segments = len(music_data)//time_segment if (len(music_data)//time_segment) == (len(music_data)/time_segment) \
                                             else 1+len(music_data)//time_segment
print(num_segments)
music_segments = np.empty([0, time_segment])
for i in range(num_segments):
    start = time_segment*(i)
    end = time_segment*(i+1)
    
    if i < num_segments-1: # 마지막 segment를 제외하면 모두 time_segment의 길이를 가짐
        ith_segment = np.expand_dims(music_data[start:end], axis=0)
    else:
        last_segment = music_data[start:]
        zero_array = np.zeros(shape=time_segment-len(last_segment))
        ith_segment = np.expand_dims(np.append(last_segment, zero_array), axis=0) # 마지막 segment를 0으로 채워 time_segment로 맞춤
    music_segments = np.append(music_segments, ith_segment, axis=0)

print(music_segments)
print(len(music_segments))

from librosa import cqt
hop_length = 512
bins_per_octave = 20
n_bins=168
time_range = 1290 # time_segment만큼의 데이터를 cqt 처리했을 때의 시간 범위
music_cqt = np.empty([0, time_range, n_bins])

for segment in music_segments:
    segment_cqt = np.abs(cqt(segment, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave))
    segment_cqt = np.expand_dims(segment_cqt.T, axis=0)
    music_cqt = np.append(music_cqt, segment_cqt, axis=0)
print(music_cqt.shape)

# 1D CNN으로 music의 장르 예측
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, AveragePooling1D
from keras.optimizers import Adagrad
from keras.utils import multi_gpu_model

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
pred = gpu_model.predict(music_cqt)

genres = ['disco', 'raggae', 'rock', 'pop', 'jazz', 'blues', 'classical', 'hiphop', 'country', 'metal']
total_pred = np.empty([0], dtype=int)
for segment in pred:
    top2 = segment.argsort()[-2:]
    total_pred = np.append(total_pred, top2)
print(total_pred)
from collections import Counter
result = Counter(total_pred).most_common(3)
first = result[0]
second = result[1]
third = result[2]
print('%d개의 segment로 나누어 상위 2개의 확률로 장르 예측' % num_segments)
print('%s : %d (%.2f%%) | %s : %d (%.2f%%) | %s : %d (%.2f%%)' % \
      (genres[first[0]], first[1], first[1]/(num_segments*2),
       genres[second[0]], second[1], second[1]/(num_segments*2),
       genres[third[0]], third[1], third[1]/(num_segments*2)))

def load_mp3(music): # music : 장르를 예측할 음악 파일 경로
    '''mp3파일을 읽어서 data로 변환하는 함수'''
    
    import librosa
    from pydub import AudioSegment
    import numpy as np
    
    sample_rate = AudioSegment.from_mp3(music).frame_rate # mp3 파일의 sample_rate 리턴
    music_data, sample_rate = librosa.load(music, sr=sample_rate)
    return music_data, sample_rate

def separate_music(music_data, sample_rate, time_segment=660000): # time_segment : 분리된 music data의 길이
    '''music data를 일정한 시간간격에 따라 n개의 segment로 나누는 함수'''
    
    import numpy as np
    
    num_segments = len(music_data)//time_segment if (len(music_data)//time_segment) == (len(music_data)/time_segment) \
                                                 else 1+len(music_data)//time_segment
    # num_segments : 분리 후 segment의 총 개수
    # music_data의 총 길이가 time_segment로 나누어 떨어지지 않으면 segment의 개수는 그 몫 + 1
    
    music_segments = np.empty([0, time_segment]) # segment 별로 music data를 저장할 빈 array 생성
    for i in range(num_segments):
        start = time_segment*(i)
        end = time_segment*(i+1)

        if i < num_segments-1: # 마지막 segment를 제외하면 모두 time_segment의 길이를 가짐
            ith_segment = np.expand_dims(music_data[start:end], axis=0)
        else:
            last_segment = music_data[start:]
            zero_array = np.zeros(shape=time_segment-len(last_segment))
            ith_segment = np.expand_dims(np.append(last_segment, zero_array), axis=0) # 마지막 segment를 0으로 채워 time_segment로 맞춤
        music_segments = np.append(music_segments, ith_segment, axis=0)
    print('num_segments=', num_segments, sep='')
    return music_segments # shape = (num_segments, time_segment)

def constant_q_transform(music_segments, hop_length=512, bins_per_octave=20, n_bins=168, time_range=1290):
    '''raw music data에 constant q transform(cqt)를 취하여 변환하는 함수'''
    # time_range : time_segment만큼의 데이터를 cqt 처리했을 때의 시간 범위
    
    from librosa import cqt
    import numpy as np
     
    music_cqt = np.empty([0, time_range, n_bins])

    for segment in music_segments:
        segment_cqt = np.abs(cqt(segment, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave))
        # hop_length : number of samples between successive CQT columns
        # bins_per_octave : Number of bins per octave - Keras model은 20으로 학습함
        # n_bins : Number of frequency bins, starting at fmin(default=32.70Hz) - Keras model은 168로 학습함
        segment_cqt = np.expand_dims(segment_cqt.T, axis=0) # n_bins를 채널로 하기위해 전치
        music_cqt = np.append(music_cqt, segment_cqt, axis=0)
    return music_cqt # shape = (num_segments, time_range, n_bins)

def predict_genre(music_cqt):
    '''훈련된 모델(1D CNN)로 음악의 장르를 예측하는 함수'''
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout
    from keras.layers.convolutional import Conv1D, AveragePooling1D
    from keras.optimizers import Adagrad
    from keras.utils import multi_gpu_model
    import numpy as np
        
    # 모델 훈련에 사용된 장르 리스트(one-hot-encoding 데이터의 각 열이 나타내는 장르)
    genres = ['disco', 'raggae', 'rock', 'pop', 'jazz', 'blues', 'classical', 'hiphop', 'country', 'metal']
    
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
    pred = gpu_model.predict(music_cqt) # segment 마다 genres list의 각 장르에 해당할 확률이 저장된 array - shape(music_segments, 1, 10)
    
    total_pred = np.empty([0], dtype=int) # segment 마다 확률 상위 2개의 인덱스를 저장하는 array
    for segment in pred:
        top2 = segment.argsort()[-2:]
        total_pred = np.append(total_pred, top2)

    from collections import Counter
    result = Counter(total_pred).most_common(3) # segment 마다 2개씩 추출된 예측 장르 인덱스를 장르별로 counting
    first = result[0]
    second = result[1]
    third = result[2]
    num_segments = music_cqt.shape[0]
    print('%d개의 segment로 나누어 segment당 상위 2개의 확률로 장르 예측' % num_segments)
    print('%s : %d (%.2f%%) | %s : %d (%.2f%%) | %s : %d (%.2f%%)' % \
          (genres[first[0]], first[1], first[1]/(num_segments*2),
           genres[second[0]], second[1], second[1]/(num_segments*2),
           genres[third[0]], third[1], third[1]/(num_segments*2)))
    return Counter(total_pred)

music = 'music_samples/blues/blues1.mp3'
result = genre(music)
print(result)

def genre(music):
    print()
    print('predict the genre of %s' % music)
    music_data, sample_rate = load_mp3(music)
    music_segments = separate_music(music_data, sample_rate=sample_rate)
    music_cqt = constant_q_transform(music_segments)
    total = predict_genre(music_cqt)
    return total

music = 'music_samples/blues/blues1.mp3'
total = genre(music)
print(total)



import os
sample_path = 'music_samples'
sample = [dirname for dirname in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, dirname))]
sample.remove('.ipynb_checkpoints')
print(sample)

import numpy as np
import librosa

# 장르를 key로, 장르별 audio track을 요소로하는 리스트가 value인 dictionary 형태의 sampleset 생성
sampleset = {}
for dirname in sample:
    dir_path = os.path.join(sample_path, dirname)
    sampleset.setdefault(dirname, [])
    for filepath in os.listdir(dir_path):
        music_sample = os.path.join(dir_path, filepath)
        sampleset[dirname].append(music_sample)
print(sampleset)

for music in sampleset['jazz']:
    genre(music)

# classical을 잘 못맞춤 -> 샘플이 모던 클래식이라 그런듯?

total = genre('music_samples/classical/classical2.mp3')

print(total)
