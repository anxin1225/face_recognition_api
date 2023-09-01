import face_recognition
import os
import math

standard_ear_threshold = 0.25
standard_mar_threshold = 0.65
standard_turn_head_threshold = 0.6

def get_turn_head_value(face_landmarks):
    chin = face_landmarks['chin']
    nose_bridge = face_landmarks['nose_bridge']

    rp = chin[0]
    lp = chin[len(chin)-1]

    np = nose_bridge[len(nose_bridge)-1]

    ld = _get_distance(lp, np)
    rd = _get_distance(rp, np)

    return (ld - rd) / min(ld, rd)

def get_turn_head_type(face_landmarks, turn_threshold):
    return get_turn_head_type_with_value(get_turn_head_value(face_landmarks), turn_threshold)

def get_turn_head_type_with_value(value, turn_threshold):
    if is_turn_head_left_with_value(value, turn_threshold):
        return "left"
    if is_turn_head_right_with_value(value, turn_threshold):
        return "right"
    
    return 'none'

def is_turn_head_left_with_value(value, turn_threshold):
    if value <= -turn_threshold:
        return True

    return False

def is_turn_head_left(face_landmarks, turn_threshold):
    return is_turn_head_left_with_value(get_turn_head_value(face_landmarks), turn_threshold)

def is_turn_head_right_with_value(value, turn_threshold):
    if value >= turn_threshold:
        return True

    return False

def is_turn_head_right(face_landmarks, turn_threshold):
    return is_turn_head_right_with_value(get_turn_head_value(face_landmarks), turn_threshold)

def _get_distance(p1, p2):
    return abs(math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))

def is_eye_open(face_landmarks, ear_threshold):
    return is_left_eye_open(face_landmarks, ear_threshold) and is_right_eye_open(face_landmarks, ear_threshold)

def is_left_eye_open(face_landmarks, ear_threshold):
    eye = face_landmarks['left_eye']
    ear_value = get_ear_value(eye)

    if ear_value > ear_threshold:
        return True
    
    return False

def is_right_eye_open(face_landmarks, ear_threshold):
    eye = face_landmarks['right_eye']
    ear_value = get_ear_value(eye)

    if ear_value > ear_threshold:
        return True

    return False


def get_ear_value(eye):
    A = _get_distance(eye[1], eye[5])
    B = _get_distance(eye[2], eye[4])

    C = _get_distance(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def is_mouth_open(face_landmarks, mar_threshold):
    mouth = face_landmarks['mouth']
    if get_mar_value(mouth) >= mar_threshold:
        return True
    
    return False


def get_mar_value(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = _get_distance(mouth[2], mouth[10]) # 51, 59
    B = _get_distance(mouth[4], mouth[8]) # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = _get_distance(mouth[0], mouth[6]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar

def face_landmarks(face_image, face_locations=None, model="large"):
    landmarks = face_recognition.face_landmarks(face_image, face_locations, model)

    for landmark in landmarks:

        tl = landmark['top_lip']
        bl = landmark['bottom_lip']

        landmark['mouth'] = tl[0:6] + bl[0:6] + [tl[11], tl[10], tl[9], tl[8], tl[7], bl[10], bl[9], bl[8]]

    return landmarks

def find_main_face(face_locations):
    max_area = 0
    max_face = face_locations[0]
    max_face_location = []
    for face in face_locations:
        area = abs((face[0] - face[2]) * (face[1] - face[3]))
        if area > max_area:
            max_area = area
            max_face = face

    max_face_location.append(max_face)

    return max_face_location    