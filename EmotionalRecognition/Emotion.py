import tensorflow as tf
import cv2
import numpy as np

data_directory = "../Upload/"
new_model = tf.keras.models.load_model('EmotionalRecognition/mod_my_model_94p69.h5')


def get_emotion(image_name):
    image_dir = data_directory + image_name
    print(image_dir)
    frame = cv2.imread(data_directory + image_name)
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
            return return_str
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]
    if len(face_roi) == 0:
        return_str = '{ "class" : -1 }'
        return return_str
    else:
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        predictions = new_model.predict(final_image)

        print(predictions[0])
        return_str = '{ "class" : ' + np.argmax(predictions) + ' }'
        return return_str


print(get_emotion('bg.jpeg'))