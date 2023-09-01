import face_recognition
from flask import Flask, jsonify, request, redirect
import json
import numpy as np
import sub_functions

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
           return analysis_faces_in_image(file)
        else:
            return '{"error":"Invalid image file format."}'

    return '''
    <!doctype html>
    <title>Get face encodings</title>
    <h1>Upload a picture and Get face encodings!</h1>
    <form method="POST" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Upload">
    </form>
    '''

@app.route('/standard_threshold', methods=['GET'])
def standard_threshold():
    result = {
        "standard_ear_threshold": sub_functions.standard_ear_threshold,
        "standard_mar_threshold": sub_functions.standard_mar_threshold,
        "standard_turn_head_left_threshold": -sub_functions.standard_turn_head_threshold,
        "standard_turn_head_right_threshold": sub_functions.standard_turn_head_threshold,
    }

    return jsonify(result)

@app.route('/match', methods=['GET', 'POST'])
def face_match():
    if request.is_json != True:
        return '{"error":"Invalid json format."}'
    
    json_data = request.get_json(cache=True)
    check_faces = json_data.get('check_faces', None)
    target_face = json_data.get('target_face', None)

    if check_faces == None or len(check_faces) <= 0:
        return '{"error":"Invalid check_faces"}'

    if target_face == None or len(target_face) <= 0:
        return '{"error":"Invalid target_face"}'

    result = face_recognition.face_distance(np.asarray(check_faces), target_face)

    #return json.dumps(result, cls=NumpyArrayEncoder)
    return jsonify(result.tolist())

def analysis_faces_in_image(file_stream):
    img = face_recognition.load_image_file(file_stream)
    
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    face_landmarks = sub_functions.face_landmarks(img, face_locations)

    faces = []

    for (top, right, bottom, left), face_encoding, face_landmark in zip(face_locations, face_encodings, face_landmarks):
        faces.append({
            "features":{
                "head_turn": sub_functions.get_turn_head_value(face_landmark),
                "eye_left_open": sub_functions.get_ear_value(face_landmark['left_eye']),
                "eye_right_open": sub_functions.get_ear_value(face_landmark['right_eye']),
                "mouth_open": sub_functions.get_mar_value(face_landmark['mouth']),
            },
            "rect":{
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
            },
            "encodings": face_encoding.tolist()
        })

    result = {
        "faces": faces
    }

    #return json.dumps(result, cls=NumpyArrayEncoder)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)


