import face_recognition
import numpy as np
import cv2

pic = ["nice.jpg","nit.png","mook.jpg"]

person1_image = face_recognition.load_image_file(pic[0])
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]


person2_image = face_recognition.load_image_file(pic[1])
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file(pic[2])
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]


known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding,
]

face_locations = []
face_encodings = []
process_this_frame = True

image1 = np.zeros((200,400,3), dtype='uint8');
temp = cv2.imread("blank.png")
temp = cv2.resize(temp,(200,200))
image1[:200,:200] = temp
images = {"nice.jpg": cv2.resize(cv2.imread(pic[0]), (200, 200)),
          "nit.png": cv2.resize(cv2.imread(pic[1]),(200,200)),
          "mook.jpg": cv2.resize(cv2.imread(pic[2]),(200,200)),
          "blank.png": temp,
          }
while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read();
    frame = cv2.resize(frame,(200,200), fx=0.25, fy=0.25)
    image1[:200,200:400] = frame
    rgb_small_frame = frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
            name = "blank.png"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            #print(best_match_index)
            if matches[best_match_index]:
                name = pic[best_match_index]
            image1[:200,:200] = images[name]
    cv2.imshow("Show",image1);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#video_capture.release()
cv2.destroyAllWindows()
