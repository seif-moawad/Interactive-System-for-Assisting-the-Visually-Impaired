import face_recognition
import numpy as np
from IPython.display import display
import dlib
import numpy as np
from PIL import Image, ImageDraw
from deepface import DeepFace
import datetime
import cv2
from gtts import gTTS
from playsound import playsound
import threading
import keyboard
from speech_recognition import Recognizer, Microphone

# Load known face encodings and their names
obama_image = face_recognition.load_image_file(r'C:\Users\seifm.DESKTOP-2G82R9G\Pictures\Screenshot 2023-05-18 230456.png')
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

seif_image = face_recognition.load_image_file(r'C:\Users\seifm.DESKTOP-2G82R9G\Pictures\Screenshot 2023-05-19 182558.png')
seif_face_encoding = face_recognition.face_encodings(seif_image)[0]

known_face_encodings = [
    obama_face_encoding,
    seif_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Seif Moawad"
]

# Initialize the webcam
video_capture = cv2.VideoCapture(1)

# Initialize the face detector and face recognition models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'G:\HCI\Project codes\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(r'G:\HCI\Project codes\dlib_face_recognition_resnet_model_v1.dat')

# Initialize the speech recognition objects
recognizer = Recognizer()
microphone = Microphone()

import tempfile
# Function to play speech
def play_speech(name):
    tts = gTTS(text=name, lang="en")
    speech_file = tempfile.mktemp(suffix=".mp3")
    tts.save(speech_file)
    playsound(speech_file)

# Function to get user response through speech input
def get_user_response():
    print("Speech: Who is this?")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        user_response = recognizer.recognize_google(audio)
        if user_response:
            print("User Response:", user_response)
            return user_response
        else:
            return None
    except:
        return None

# Variables to track detected names and played speeches
detected_names = []
speech_played = False

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    face_rects = detector(gray_frame, 0)

    # Create a PIL ImageDraw Draw instance to draw on top of the frame
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the frame
    for face_rect in face_rects:
        # Detect facial landmarks
        shape = predictor(gray_frame, face_rect)
        face_encoding = np.array(facerec.compute_face_descriptor(frame, shape))

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name not in detected_names and not speech_played:
            # Play speech with detected name
            thread_play_speech = threading.Thread(target=play_speech, args=(name,))
            thread_play_speech.start()

            # Add the name to the detected names list
            detected_names.append(name)
            # Set the speech_played flag to True
            speech_played = True

        elif name == "Unknown" and not speech_played:
            # Play speech with "Unknown"
            thread_play_speech = threading.Thread(target=play_speech, args=(name,))
            thread_play_speech.start()

            # Set the speech_played flag to True
            speech_played = True
            # Get user response to add a name
            user_response = get_user_response()
            if user_response:
                name = user_response
                # Add the name to the known_face_names list
                known_face_names.append(name)
                # Set the speech_played flag to False to play speech for the added name
                speech_played = False
                # Play speech with the added name
                play_speech(name)


        # Draw a rectangle around the face
        top, right, bottom, left = face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left()
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with the name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        
    # Remove the drawing library from memory
    del draw

    # Convert the PIL image back to BGR format for display
    frame = np.array(pil_image)[:, :, ::-1]

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Analyze facial features and save data
if name != "Unknown":
    # Analyze facial features
    obj = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'])
    result = obj[0]
    # Access individual attributes
    age = result["age"]
    gender = result["gender"]
    dominant_race = result["dominant_race"]
    dominant_emotion = result["dominant_emotion"]

    # Printing the results
    print(age, " years old ", dominant_race, " ", dominant_emotion, " ", gender)

    # Formatting the data string
    data_string = f"{name},{age},{dominant_race},{dominant_emotion},{gender},{datetime.datetime.now()},{'$'}\n"

    # Path to the output text file
    output_file = r'path_to_output_file.txt'

    # Saving the data to the text file
    with open(output_file, "a") as file:
        file.write(data_string)

    # Printing the results
    print("Data saved successfully.")
