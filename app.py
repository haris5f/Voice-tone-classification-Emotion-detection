import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import pickle
from keras.preprocessing.sequence import pad_sequences
import librosa
from scipy.io import wavfile


app = Flask(__name__)
model = keras.models.load_model('model_sentiment')
token = pickle.load(open('token_.pkl', 'rb'))

model_audio = keras.models.load_model('model_tone_classificatiom')
sc = pickle.load(open('sc.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text = token.texts_to_sequences([message])
        text = pad_sequences(text, maxlen=18, dtype='int32', value=0)
        res = model.predict(text, batch_size=1,verbose = 1)
        if np.argmax(res) == 0:output = "Negative"
        elif np.argmax(res) == 1:output = "Neutral"
        else: output = "Postitive"

    return render_template('index.html', prediction_text='Sentiment: {}'.format(output))

@app.route('/audioinput', methods = ['GET', 'POST'])
def upload_file():
    def extract_feature(x,sr):
        X=[] # feature vector
        mfccs = librosa.feature.mfcc(x, sr=sr)
        for i in mfccs:
            X.extend([np.mean(i)])  
        return X
    if request.method == 'POST':
        f = request.files["audiofile"]
        if f:
            x_t,sr_t = librosa.load(f,duration=3, offset=0, res_type='kaiser_fast')  # read audio file
            #sr_t, x_t = wavfile.read('output.wav')
            x_t = extract_feature(x_t,sr_t)
            x_t =np.array(x_t)
            x_t = sc.transform(x_t.reshape(1, -1))
            labels_dict ={0:'neutral',1:'calm',2:'happy',3:'sad',4:'angry',5:'fearful',6:'disgust',7:'surprised'}
            outp = labels_dict.get(np.argmax(model_audio.predict(x_t)))
        else: outp = 'no file uploaded'
        
    return render_template('index.html', prediction_audio='Sentiment: {}'.format(outp))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
    #app.run(debug=True)