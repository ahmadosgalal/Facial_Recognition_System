import face_recognition
import cv2


class FaceDetectorReady:
    def __init__(self):
        self.frame_resizing = 1

    def detect_borders(self, img):
        resized_img = cv2.resize(img, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        for face_loc in face_locations:
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(resized_img, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 200), 4)
        cv2.imshow("Frame", resized_img)


class FaceDetectorbyHand:
    def __init__(self):
        self.frame_resizing = 1

    def detect_borders(self, img):
        resized_img = cv2.resize(img, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        for face_loc in face_locations:
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(resized_img, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 200), 4)
        cv2.imshow("Frame", resized_img)
