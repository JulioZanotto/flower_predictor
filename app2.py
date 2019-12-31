import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import torch
from torch import nn
import numpy as np
import json
import matplotlib.pyplot as plt
import torchvision.models as models
from glob import glob
import shutil

UPLOAD_FOLDER = './static'

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

with open('class_idx.json', 'r') as f:
    class_to_idx = json.load(f)

# Visualize some sample data
classes = cat_to_name


def acha_nome(x):
    for flor, number in class_to_idx.items():
        if number == x:
            break
    return flor


train_on_gpu = torch.cuda.is_available()


def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    pretrained_model = getattr(models, chpt['arch'])
    if callable(pretrained_model):
        model = pretrained_model(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture not recognized")

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                             nn.ReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(512, 102))

    # Put the classifier on the pretrained network
    model.load_state_dict(chpt['model_state_dict'])

    return model


model = load_model('model_flower.pt')
model.to('cpu')


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img


def predict(image_path, model, top_num=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()

    # TODO: Implement the code to predict the class from an image file
    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    image_tensor.to('cpu')
    model_input.to('cpu')
    model.to('cpu')

    # Probs
    probs = torch.exp(model(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[acha_nome(lab)] for lab in top_labs]
    return top_probs, top_labels, top_flowers


def plot_bar_x(image_path, model):

    num = np.random.normal()
    # this is for plotting purpose
    plt.figure(figsize=(15, 4))

    # Make prediction
    probs, labs, flowers = predict(image_path, model)
    flowers = flowers[::-1]
    probs = probs[::-1]
    # Plot bar chart
    plt.barh(flowers, probs)
    plt.title('Prediction')
    plt.savefig('./static/' + str(num) + 'pred.jpg')
    plt.close()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/index')
def show_index():
    file_ = glob('./static/*.jpg')
    plot_bar_x(file_[0], model)
    for file in file_:
        os.remove(file)
    file2 = glob('./static/*.jpg')
    print(file2)
    full_filename = file2[0]
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'pred.jpg')
    return render_template("index.html", image_name=full_filename)


@app.route('/index', methods=['POST'])
def contact():
    if request.method == 'POST':
        if request.form['submit_button'] == 'home':
            return redirect('/')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/index')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
