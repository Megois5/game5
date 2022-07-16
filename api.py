from flask import Flask, request, url_for, redirect, render_template
from flask_cors import CORS
import json
import werkzeug
import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
import colorama
from pycaret.classification import *
import epitran

epi = epitran.Epitran('sin-Sinh')

colorama.init()
from colorama import Fore, Style, Back
import pickle

with open("PersonalAssistant/intents.json") as file:
    data = json.load(file)
model = keras.models.load_model('PersonalAssistant/chat_model')

# load tokenizer object
with open('PersonalAssistant/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

    # load label encoder object
with open('PersonalAssistant/label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# old models
Decision_Tree_Classifier = load_model('Classification/trained_models/Decision_Tree_Classifier')  # best 1
Ridge_Classifier = load_model('Classification/trained_models/Ridge_Classifier')  # best 2
SVM_Linear_Kernel = load_model('Classification/trained_models/SVM_Linear_Kernel')  # best 3

# new models
model1 = load_model('Classification/new_train_models/Ada_Boost_Classifier')
model2 = load_model('Classification/new_train_models/Gradient_Boosting_Classifier')
model3 = load_model('Classification/new_train_models/Logistic_Regression')

# answer models
ans_model1 = load_model('Classification/answer_train_models/Extra_Trees_Classifier')
ans_model2 = load_model('Classification/answer_train_models/Gradient_Boosting_Classifier')
ans_model3 = load_model('Classification/answer_train_models/Random_Forest_Classifier')

app = Flask(__name__)
CORS(app)

data_directory = "Upload/"
new_model = tf.keras.models.load_model('EmotionalRecognition/mod_my_model_94p69.h5')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    input = request.form['input']
    max_len = 20

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([input]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    return_str = '{ "response" : "' + str("did not respond") + '" }'

    for i in data['intents']:
        if i['tag'] == tag:
            return_str = '{ "response" : "' + str(np.random.choice(i['responses'])) + '" }'

    return json.loads(return_str)


@app.route('/question_classification', methods=['GET', 'POST'])
def question_classification():
    question = request.form['question']

    input2 = epi.transliterate(question)
    data = np.array([['Question'], [input2]])
    result2 = predict_model(model3, data=pd.DataFrame(data=data[0, 0], index=data[0:, 0], columns=data[0, 0:])).iat[
        1, 1]

    return_str = '{ "category" : "' + str(result2) + '" }'

    return json.loads(return_str)


@app.route('/answer_classification', methods=['GET', 'POST'])
def answer_classification():
    cat_1 = request.form['cat_1']
    cat_2 = request.form['cat_2']
    cat_3 = request.form['cat_3']

    data = np.array([['cat_01', 'cat_02', 'cat_03'], [cat_1, cat_2, cat_3]])

    result = \
    predict_model(ans_model1, data=pd.DataFrame(data=data[0:, 0:], index=data[0:, 0], columns=data[0, 0:])).iat[1, 4]

    return_str = '{ "level" : ' + str(result) + ' }'

    return json.loads(return_str)


@app.route('/emotion', methods=['GET', 'POST'])
def emotion():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('Upload/' + filename)

    image_dir = data_directory + filename
    print(image_dir)
    frame = cv2.imread(data_directory + filename)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_roi = []
    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        facess = face_cascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            return_str = '{ "class" : -1 }'
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]
    if len(face_roi) == 0:
        return_str = '{ "class" : -1 }'
    else:
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        predictions = new_model.predict(final_image)

        print(predictions[0])
        return_str = '{ "class" : ' + str(np.argmax(predictions)) + ' }'

    return json.loads(return_str)


if __name__ == '__main__':
    app.run(host="192.168.1.5", port=2211, debug=True)
