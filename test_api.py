from flask import Flask, request, url_for, redirect, render_template
from flask_cors import CORS
import json
import werkzeug

app = Flask(__name__)
CORS(app)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return_str = '{ "response" : "' + str("chat bot response") + '" }'

    return json.loads(return_str)


@app.route('/question_classification', methods=['GET', 'POST'])
def question_classification():
    question = request.json['question']
    return_str = '{ "category" : "' + str("numeric") + '" }'

    return json.loads(return_str)


@app.route('/answer_classification', methods=['GET', 'POST'])
def answer_classification():
    answer = request.json['answer']
    return_str = '{ "accuracy" : ' + str("1") + ' }'

    return json.loads(return_str)


@app.route('/emotion', methods=['GET', 'POST'])
def emotion():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    return_str = '{ "category" : ' + str("1") + ' }'

    return json.loads(return_str)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000, debug=True)
