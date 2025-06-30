import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folders and set permissions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
os.chmod(RESULT_FOLDER, 0o777)

# Load model
model = YOLO('best.onnx')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # Save uploaded file
                file.save(upload_path)
                
                # Run detection
                results = model.predict(upload_path, conf=0.5)
                res_img = results[0].plot()  # Get annotated image
                
                # Save result
                result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
                cv2.imwrite(result_path, res_img)
                
                # Verify files exist
                if not os.path.exists(upload_path) or not os.path.exists(result_path):
                    raise Exception("Failed to save images")
                
                return render_template('index.html', 
                                    upload=f"uploads/{filename}",
                                    result=f"results/{filename}")
            
            except Exception as e:
                return render_template('index.html', error=f"Error: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)