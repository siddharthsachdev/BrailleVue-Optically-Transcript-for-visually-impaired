from flask import Flask, render_template, request, Response, session

from flask import send_from_directory
from flask import url_for

from werkzeug.utils import secure_filename
import functools
import operator
import os
import cv2
import time
import pyttsx3
import joblib
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import extract_features
from threading import Thread
import datetime, time

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

UPLOAD_FOLDER = 'data/testing_data/video/'
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
camera = cv2.VideoCapture(0)


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        video_to_text = VideoDescriptionRealTime(config)
        video_to_text.load_inference_models()

        start = time.time()
        video_caption, file = video_to_text.test(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        end = time.time()

        sentence = ' '.join(video_caption.split())

        print(f'It took {end - start:.2f} seconds to generate caption: {sentence}')

        engine = pyttsx3.init()
        engine.say(sentence)
        engine.runAndWait()

        return render_template('result.html', sentence=sentence, filename=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            global file__name
            if (rec):
                now = datetime.datetime.now()
                import random
                import string

                file_name = ''.join(random.choices(string.ascii_lowercase, k=10))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(r'data/testing_data/video/vid_{}.avi'.format(file_name), fourcc, 20.0, (640, 480))
                file__name = 'vid_{}.avi'.format(file_name)

                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                print('else')
                time.sleep(5)
                video_to_text = VideoDescriptionRealTime(config)
                video_to_text.load_inference_models()

                start = time.time()
                video_caption, file = video_to_text.test(os.path.join(app.config['UPLOAD_FOLDER'], file__name))
                end = time.time()
                sentence = ''
                for text in video_caption.split():
                    sentence = sentence + ' ' + text
                print(sentence)

                print('It took {:.2f} seconds to generate caption'.format(end - start))

                engine = pyttsx3.init()
                engine.say(sentence)

                engine.runAndWait()
                return sentence
                out.release()


    elif request.method == 'GET':
        return render_template('upload.html')
    return render_template('upload.html')


def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (neg):
                frame = cv2.bitwise_not(frame)
            if (capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


class VideoDescriptionRealTime(object):
    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability

        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.search_type = config.search_type
        self.tokenizer = None
        self.num = 0

    def load_inference_models(self):
        # load tokenizer

        with open(self.save_model_path + '/tokenizer' + str(self.num_decoder_tokens), 'rb') as file:
            self.tokenizer = joblib.load(file)

        # inference encoder model
        self.inf_encoder_model = load_model(self.save_model_path + '/encoder_model.h5')

        # inference decoder model
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.inf_decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        self.inf_decoder_model.load_weights(self.save_model_path + '/decoder_model_weights.h5')

    def greedy_search(self, f):
        """

        :param f: the loaded numpy array after creating videos to frames and extracting features
        :return: the final sentence which has been predicted greedily
        """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(f.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, 1500))
        final_sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if inv_map[y_hat] is None:
                break
            if inv_map[y_hat] == 'eos':
                break
            else:
                final_sentence = final_sentence + inv_map[y_hat] + ' '
                target_seq = np.zeros((1, 1, 1500))
                target_seq[0, 0, y_hat] = 1
        return final_sentence

    def decode_sequence2bs(self, input_seq):
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq

    def beam_search(self, target_seq, states_value, prob, path, lens):
        """

        :param target_seq: the array that is fed into the model to predict the next word
        :param states_value: previous state that is fed into the lstm cell
        :param prob: probability of predicting a word
        :param path: list of words from each sentence
        :param lens: number of words
        :return: final sentence
        """
        global decode_seq
        node = 2
        output_tokens, h, c = self.inf_decoder_model.predict(
            [target_seq] + states_value)
        output_tokens = output_tokens.reshape(self.num_decoder_tokens)
        sampled_token_index = output_tokens.argsort()[-node:][::-1]
        states_value = [h, c]
        for i in range(node):
            if sampled_token_index[i] == 0:
                sampled_char = ''
            else:
                sampled_char = list(self.tokenizer.word_index.keys())[
                    list(self.tokenizer.word_index.values()).index(sampled_token_index[i])]
            MAX_LEN = 12
            if sampled_char != 'eos' and lens <= MAX_LEN:
                p = output_tokens[sampled_token_index[i]]
                if sampled_char == '':
                    p = 1
                prob_new = list(prob)
                prob_new.append(p)
                path_new = list(path)
                path_new.append(sampled_char)
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index[i]] = 1.
                self.beam_search(target_seq, states_value, prob_new, path_new, lens + 1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1)
                if p > self.max_probability:
                    decode_seq = path
                    self.max_probability = p

    def decoded_sentence_tuning(self, decoded_sentence):
        # tuning sentence
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ""
        for idx2, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1
            if last_string == c and idx2 > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c
        return decode_str

    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def get_test_data(self, file_name):
        path = self.test_path + '/feat/' + file_name + '.npy'
        if os.path.exists(path):
            f = np.load(path)
        else:
            model = extract_features.model_cnn_load()
            f = extract_features.extract_features(file_name, model)
        # model = extract_features.model_cnn_load()
        # f = extract_features.extract_features(file_name, model)

        return f, file_name

    def test(self, filename):
        X_test, filename = self.get_test_data(filename)
        # generate inference test outputs
        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(X_test.reshape((-1, 80, 4096)))
        else:
            sentence_predicted = ''
            decoded_sentence = self.decode_sequence2bs(X_test.reshape((-1, 80, 4096)))
            decode_str = self.decoded_sentence_tuning(decoded_sentence)
            for d in decode_str:
                sentence_predicted = sentence_predicted + d + ' '
        # re-init max prob
        self.max_probability = -1
        return sentence_predicted, filename


if __name__ == '__main__':
    app.run(debug=True)