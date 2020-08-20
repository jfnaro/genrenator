from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, url_for
from sklearn.preprocessing import normalize
import numpy as np
import librosa as lr
import pandas as pd

app = Flask(__name__)

GENRES = [  'Hip hop',
            'Reggae',
            'Pop',
            'Blues',
            'Country',
            'Jazz',
            'Metal',
            'Disco',
            'Classical',
            'Rock']

class song_analysis():

    def __init__(self):
        self.title = ''
        self.song_analysis = []

@app.route('/', methods=['GET','POST'])
def  handle_song_upload():

    if request.method == 'GET':
        return render_template('index.html')

    song_info = song_analysis()
    song = request.files['song']
    song_title = song.filename

    if song_title.strip() == '':
        return render_template('index.html')

    if not song_title.endswith('.wav'):
        return render_template('results.html', song_results={'title':'Incorrect file format. Only .wav files are allowed.'})

    song_info.title = song_title

    #--------------------------------------preprocess data-----------------------------------------
    
    temp = open('temp.wav', 'wb')
    temp.write(song.read())
    data, sample_rate = lr.load('temp.wav')

    metadata = pd.read_csv('min_dif.csv')

    sample_size = 30 * sample_rate
    features = []
    genres = []

    model_1 = load_model('best_model_relu.h5')

    for index in range(0, len(data)-int(sample_size/2), int(sample_size/2)):

        cur_data = data[index:index+(sample_size)]
        
        c_s = lr.feature.chroma_stft(cur_data)
        features.append(np.mean(c_s))
        features.append(np.var(c_s))
        
        rms = lr.feature.rms(cur_data)
        features.append(np.mean(rms))
        features.append(np.var(rms))
        
        s_c = lr.feature.spectral_centroid(cur_data)
        features.append(np.mean(s_c))
        features.append(np.var(s_c))
        
        s_b = lr.feature.spectral_bandwidth(cur_data)
        features.append(np.mean(s_b))
        features.append(np.var(s_b))
        
        s_r = lr.feature.spectral_rolloff(cur_data)
        features.append(np.mean(s_r))
        features.append(np.var(s_r))
        
        z_c_r = lr.feature.zero_crossing_rate(cur_data)
        features.append(np.mean(z_c_r))
        features.append(np.var(z_c_r))
        
        harm = lr.effects.harmonic(cur_data)
        features.append(np.mean(harm))
        features.append(np.var(harm))
        
        features.append(lr.beat.tempo(cur_data))
        
        for n in range(1, 21):
            mfcc = lr.feature.mfcc(cur_data)
            features.append(np.mean(mfcc))
            features.append(np.var(mfcc))
        
        i = 0
        for row in metadata.iterrows():
            features[i] = (features[i] - row[1]['min']) / row[1]['dif']
            i += 1
        
        features = np.array(features, dtype=np.float)
        
        genres.append(model_1.predict(np.expand_dims(features,0)))
        
        features = []
        
    genres = np.mean(np.squeeze(genres), axis=0)

    genre_stats = []

    for value, index in sorted(((value, index) for index, value in enumerate(genres)), reverse=True):
        if value > 0.09 or len(genres) == 0:
            genre_stats.append({'genre':GENRES[index], 'percentage':value})

    #--------------------------------generate and send page----------------------------------------
    song_info.song_analysis = genre_stats

    return render_template('results.html', song_results=song_info)
