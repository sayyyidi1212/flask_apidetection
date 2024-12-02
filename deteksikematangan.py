from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("C:/Users/sadeg/dataset2/model_saya.h5")

# Konfigurasi folder upload
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variabel global untuk menyimpan prediksi terakhir
latest_prediction = None
latest_confidence = None

def allowed_file(filename):
    """Validasi apakah file memiliki ekstensi yang diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk memproses gambar dari web.
    """
    global latest_prediction, latest_confidence  # Akses variabel global untuk menyimpan prediksi terakhir
    
    # Cek apakah ada file yang diupload
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = datetime.now().strftime("%d%m%y-%H%M%S") + ".png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Buat folder upload jika belum ada
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Simpan file gambar
        file.save(filepath)

        try:
            # Proses gambar
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Prediksi dengan model
            predictions = model.predict(img_array)[0]
            class_names = ['matang', 'mentah', 'setengah_matang']
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100

            # Simpan prediksi terakhir
            latest_prediction = class_names[predicted_class]
            latest_confidence = confidence

            # Kirim respon ke web
            return jsonify({
                'filename': filepath,
                'prediction': latest_prediction,
                'confidence': f'{confidence:.2f}%'
            })

        except Exception as e:
            return jsonify({'error': f'Error processing image: {e}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_status', methods=['GET'])
def predict_status():
    """
    Endpoint untuk ESP8266 membaca prediksi terakhir.
    """
    global latest_prediction, latest_confidence
    if latest_prediction is not None:
        return jsonify({
            'prediction': latest_prediction,
            'confidence': f'{latest_confidence:.2f}%'
        }), 200
    return jsonify({'error': 'No prediction yet'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
