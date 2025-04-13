import math
from PIL import Image
import numpy as np
from flask import Flask, request, send_file, render_template_string
import os
import tempfile

app = Flask(__name__)

def exe_to_image(byte_data):
    # Calculate dimensions for a square-ish image
    data_len = len(byte_data)
    width = math.ceil(math.sqrt(data_len))
    height = width
    
    # Create a numpy array and pad with zeros if needed
    img_array = np.zeros((height, width), dtype=np.uint8)
    np_data = np.frombuffer(byte_data, dtype=np.uint8)
    img_array.flat[:len(np_data)] = np_data
    
    # Create the image
    img = Image.fromarray(img_array, mode='L')
    
    # Save the image to a temporary file
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, 'output.png')
    img.save(output_path)
    
    return output_path

@app.route('/mlReport', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return "No selected file", 400
        
        # Ensure the uploaded file has the correct extension
        if not file.filename.endswith('.exe'):
            return "Uploaded file must be an .exe file", 400
        
        # Read the file's binary data
        byte_data = file.read()
        
        # Convert the executable to an image
        output_path = exe_to_image(byte_data)
        
        # Send the generated image back to the user
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output.png')
    
    # Render the upload form for GET requests
    return render_template_string('''
        <!doctype html>
        <html>
            <head>
                <title>Upload .exe File</title>
            </head>
            <body>
                <h1>Upload an .exe File</h1>
                <form method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".exe">
                    <button type="submit">Convert to Image</button>
                </form>
            </body>
        </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)