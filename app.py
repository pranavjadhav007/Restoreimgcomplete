from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(main_image_path, mask_image_path):
    image = cv2.imread(main_image_path)
    marked_damages = cv2.imread(mask_image_path, 0)
    ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(thresh1, kernel, iterations=1)
    processed_mask_filename = "uploads/processed_mask.png"
    cv2.imwrite(processed_mask_filename, mask)
    restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    final_mask_filename = "static/final_mask.png"
    cv2.imwrite(final_mask_filename, restored)

    return processed_mask_filename, final_mask_filename

@app.route('/')
def index():
    return render_template('index1.html', final_mask=None)

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        main_image_file = request.files['file']
        main_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(main_image_file.filename))
        main_image_file.save(main_image_path)

        mask_image_file = request.files['mask']
        mask_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(mask_image_file.filename))
        mask_image_file.save(mask_image_path)

        processed_mask, final_mask = model_predict(main_image_path, mask_image_path)

        return jsonify(final_mask_image="static/final_mask.png")

if __name__ == '__main__':
    app.run(debug=True)
