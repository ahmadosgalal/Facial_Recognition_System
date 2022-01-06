import cv2
import face_recognition
from LBPH import *
from face_detector import *
from plot_local_binary_pattern import *
import skimage.feature as ft


img = cv2.imread("images/img3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("images/img1.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread("images/Ryan Reynolds.jpg")
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

flag = input("Choose '1' for training & '2 for testing: ")
if flag == '1':
    flag = input("Choose '1' for first time & '2 for adding an image: ")
    if flag == '1':
        enc_list = []
        enc_list.append(("Ant-Man", ft.local_binary_pattern(img, 8, 1, 'uniform')))
        enc_list.append(("Natasha", ft.local_binary_pattern(img2, 8, 1, 'uniform')))
        weights_array = np.array(enc_list, dtype=object)
        with open('weights.npy', 'wb') as f:
            np.save(f, weights_array)

    elif flag == '2':
        with open('weights.npy', 'rb') as f:
            weights_array = np.load(f, allow_pickle=True)
        enc_list = weights_array.tolist()
        enc_list.append(("Ryan", ft.local_binary_pattern(img3, 8, 1, 'uniform')))
        weights_array = np.array(enc_list, dtype=object)
        with open('weights.npy', 'wb') as f:
            np.save(f, weights_array)

elif flag == '2':
    with open('weights.npy', 'rb') as f:
        weights_array = np.load(f, allow_pickle=True)
    print(weights_array)
    face_detect = FaceDetectorReady()
    lbph = LBPReady(8, 1)
    classifier = Matcher(8, 1)
    n, s = classifier.match(weights_array, img3)
    print("#Main\n", n, s)





# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect face position
#face_locations = face_detect.detect_borders(img)
#lbph_hist = lbph.compute(img)


# Use my encodings
# img_encoding = face_recognition.face_encodings(rgb_img)[0]

# img2 = cv2.imread("images/Messi.webp")
# rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# Detect face position
# Use my encodings
# img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

# result = face_recognition.compare_faces([img_encoding], img_encoding2)
# print("Result: ", result)

# cv2.imshow("Img", img)
# cv2.imshow("Img 2", img2)
cv2.waitKey(0)
