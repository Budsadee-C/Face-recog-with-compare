import face_recognition
import numpy as np
import cv2

person1_image = face_recognition.load_image_file("xxx.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("xxx1.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding
]

face_locations = [];
face_encodings = [];
face_names = [];
process_this_frame = True;

image1 = np.zeros((200,400,3), dtype='uint8');
temp = cv2.imread("blank.png");
temp = cv2.resize(temp,(200,200));
image1[:200,:200] = temp;
#print(image1.shape);

while True:
    cap = cv2.VideoCapture(0);
    ret, frame = cap.read();
    frame = cv2.resize(frame,(200,200), fx=0.25, fy=0.25);
    image1[:200,200:400] = frame;
    cv2.imshow("Show",image1);


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
video_capture.release()
cv2.destroyAllWindows()
