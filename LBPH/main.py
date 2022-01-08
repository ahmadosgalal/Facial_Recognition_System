import cv2
import face_recognition
import os
from LBPH import *
from face_detector import *
from plot_local_binary_pattern import *
import skimage.feature as ft

# path = "images"
# dir_list = os.listdir(path)
#
# for item in dir_list:
#     img_name = item.split(".")[0]
#     print(img_name)

face_detect = FaceDetectorReady()
lbph = LBPReady(8, 1)
#lbph_2 = LBPbyHand(8, 1)
lbph_2 = LBPHfromScratch()
classifier = Matcher(8, 1)

flag = input("Choose '1' for training & '2 for testing: ")
if flag == '1':
    flag = input("Choose '1' for first time & '2 for adding an image: ")
    if flag == '1':
        path = "images"
        dir_list = os.listdir(path)
        enc_list = []
        img_list = []

        for item in dir_list:
            img_name = item.split(".")[0]
            print(img_name)
            img_curr = cv2.imread(path + "/" + item)

            cropped_img, face_loc_img = face_detect.detect_borders(img_curr)
            if cropped_img is None:
                continue
            #print(cropped_img)
            for cropped_img_item in cropped_img:
                gray_img = cv2.cvtColor(cropped_img_item, cv2.COLOR_RGB2GRAY)
                temp_img = lbph_2.Compute_LBP(gray_img)
                #enc_list.append((img_name, ft.local_binary_pattern(gray_img, 8, 1, 'uniform')))
                #cv2.imshow(img_name, lbph_2.Compute_LBP(gray_img))
                # Saving the image
                cv2.imwrite("mine/"+img_name+".jpg", temp_img)
                enc_list.append((img_name, temp_img))

        weights_array = np.array(enc_list, dtype=object)

        with open('weights.npy', 'wb') as f:
            np.save(f, weights_array)

    elif flag == '2':
        path = input("Enter image path: ")
        path_arr = path.split("/")
        im_name = path_arr[-1].split(".")[0]
        try:
            with open('weights.npy', 'rb') as f:
                weights_array = np.load(f, allow_pickle=True)
            enc_list = weights_array.tolist()

            img3 = cv2.imread(path)
            cropped_img3, face_loc_img3 = face_detect.detect_borders(img3)
            #print(cropped_img3)
            for cropped_img_item in cropped_img3:
                gray_img3 = cv2.cvtColor(cropped_img_item, cv2.COLOR_RGB2GRAY)
                enc_list.append((im_name, lbph_2.Compute_LBP(gray_img3)))


            #enc_list.append((im_name, ft.local_binary_pattern(gray_img3, 8, 1, 'uniform')))

            weights_array = np.array(enc_list, dtype=object)
            with open('weights.npy', 'wb') as f:
                np.save(f, weights_array)
        except Exception as E:
            print(E)
elif flag == '2':
    cap = cv2.VideoCapture(0)
    with open('weights.npy', 'rb') as f:
        weights_array = np.load(f, allow_pickle=True)
    while True:
        ret, frame = cap.read()

        # Detect Faces
        try:
            cropped_test_img, face_loc_test_img = face_detect.detect_borders(frame)

            #print("#Main\n", n, s)
            #face_locations, face_names = sfr.detect_known_faces(frame)
            for cropped_img_item, face_loc in zip(cropped_test_img,face_loc_test_img):
                gray_test_img = cv2.cvtColor(cropped_img_item, cv2.COLOR_RGB2GRAY)
                lbp_img = lbph_2.Compute_LBP(gray_test_img)
                n, s = classifier.match(weights_array, lbp_img)

                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, n+" "+str(s), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        except Exception as e:
            print(e)

        cv2.imshow("User", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    #test_img = cv2.imread("Messi1.webp")

    # with open('weights.npy', 'rb') as f:
    #     weights_array = np.load(f, allow_pickle=True)
    #print(weights_array)
    # cropped_test_img, face_loc_test_img = face_detect.detect_borders(test_img)
    # gray_test_img = cv2.cvtColor(cropped_test_img, cv2.COLOR_RGB2GRAY)

    # n, s = classifier.match(weights_array, gray_test_img)
    # print("#Main\n", n, s)

    # for face_loc in face_loc_test_img:
    #     y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
    #
    #     cv2.putText(test_img, n, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    #     cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 0, 200), 4)
    #
    # cv2.imshow(n, test_img)

###################################################
# img = cv2.imread("images/Messi.webp")
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img2 = cv2.imread("images/img1.png")
# #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img3 = cv2.imread("images/Ryan Reynolds.jpg")
# #img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
# test_img = cv2.imread("Messi1.webp")

# face_detect = FaceDetectorReady()
# lbph = LBPReady(8, 1)
# classifier = Matcher(8, 1)

# cropped_img, face_loc_img = face_detect.detect_borders(img)
# cropped_img2, face_loc_img2 = face_detect.detect_borders(img2)
# cropped_img3, face_loc_img3 = face_detect.detect_borders(img3)
# cropped_test_img, face_loc_test_img = face_detect.detect_borders(test_img)

# gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
# gray_img2 = cv2.cvtColor(cropped_img2, cv2.COLOR_RGB2GRAY)
# gray_img3 = cv2.cvtColor(cropped_img3, cv2.COLOR_RGB2GRAY)
# gray_test_img = cv2.cvtColor(cropped_test_img, cv2.COLOR_RGB2GRAY)

# cv2.imshow("Debug",gray_img3)

# enc_list = []
# enc_list.append(("Messi", ft.local_binary_pattern(gray_img, 8, 1, 'uniform')))
# enc_list.append(("Natasha", ft.local_binary_pattern(gray_img2, 8, 1, 'uniform')))
# enc_list.append(("Ryan", ft.local_binary_pattern(gray_img3, 8, 1, 'uniform')))
# weights_array = np.array(enc_list, dtype=object)

# n, s = classifier.match(weights_array, gray_test_img)
# print("#Main\n", n, s)
#
# for face_loc in face_loc_test_img:
#     y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
#
#
#     cv2.putText(test_img, n, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#     cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 0, 200), 4)
#
# cv2.imshow(n, test_img)


# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect face position
# face_locations = face_detect.detect_borders(img)
# lbph_hist = lbph.compute(img)


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