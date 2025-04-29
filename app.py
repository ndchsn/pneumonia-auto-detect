from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model import predict_disease

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_disease(filepath)
    return render_template('index.html', result=result, image_path=filepath)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

