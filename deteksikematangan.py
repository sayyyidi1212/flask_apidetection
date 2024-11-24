from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from datetime import datetime
from flask_cors import CORS  # Untuk mengaktifkan CORS

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS

# Load model
model = load_model("C:/Users/sadeg/dataset2/model_saya.h5")  # Pastikan path benar

# Konfigurasi folder upload
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Validasi apakah file memiliki ekstensi yang diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah ada file yang diupload
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # Pastikan file memiliki ekstensi yang valid
    if file and allowed_file(file.filename):
        print(f"File received: {file.filename}")  # Log nama file yang diterima

        # Simpan file dengan nama yang unik
        filename = datetime.now().strftime("%d%m%y-%H%M%S") + ".png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Buat folder upload jika belum ada
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Simpan file ke server
        file.save(filepath)

        # Proses gambar
        try:
            # Muat gambar dan ubah ukuran sesuai dengan model
            img = load_img(filepath, target_size=(224, 224))
            print(f"Image loaded: {filepath}")  # Log setelah gambar berhasil dimuat
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalisasi gambar

            # Lakukan prediksi menggunakan model
            predictions = model.predict(img_array)[0]
            class_names = ['matang', 'mentah', 'setengah_matang']
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100

            print(f"Prediction: {class_names[predicted_class]} with confidence {confidence:.2f}%")

            # Kembalikan hasil prediksi dalam format JSON dengan path relatif
            filename_relative = os.path.join('uploads', filename)
            return jsonify({
                'filename': filename_relative,  # Path gambar yang disimpan secara relatif
                'prediction': class_names[predicted_class],  # Kelas prediksi
                'confidence': f'{confidence:.2f}%'  # Tingkat kepercayaan
            })

        except Exception as e:
            # Tangani error saat memproses gambar
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Error processing image'}), 500

    # Jika file tidak valid
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
