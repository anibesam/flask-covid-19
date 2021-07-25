from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, redirect, templating, url_for, request, render_template
import os

from werkzeug.utils import HTMLBuilder
#from werkzeug import secure_filename
app = Flask(__name__)


@app.route('/report/<name>')
def report(name):
    # dimensions of our images
    img_width, img_height = 224, 224

    # load the model we saved
    model = load_model('keras.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # predicting images
    #img = image.load_img(name).convert('L')
    #img = img.resize(img_height, img_width)
    img = image.load_img(name, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)

    if classes[0][0] == 1:
        return render_template("negative.html")

    else:
        return render_template("positive.html")


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        file = request.files['nm']
        basepath = os.path.dirname(__file__)
        #file.save(os.path.join(basepath, "uploads", file.filename))
        #user = os.path.join(basepath, "uploads", file.filename)
        file.save(os.path.join(basepath, file.filename))
        user = file.filename
        return redirect(url_for('report', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('report', name=user))


@app.route("/") 
def home_view(): 
        return render_template("index.html")

@app.route("/app")
def app_view():
  return render_template("app.html")

@app.route("/negative")
def negative_view():
  return render_template("negative.html") 

@app.route("/positive")
def positive_view():
  return render_template("positive.html")    


if __name__ == '__main__':
    app.run(debug=True)
