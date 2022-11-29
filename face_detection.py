# Importing libs
import cv2
import numpy as npy
import face_recognition as face_rec

# functions
def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

# img declaration
shubham = face_rec.load_image_file ('sample_image\shubham.jpg')
shubham = cv2.cvtColor(shubham, cv2.COLOR_BGR2RGB)
shubham = resize(shubham, 0.50)
shubham_test = face_rec.load_image_file ('sample_image\shubham_test.jpg')
shubham_test = cv2.cvtColor(shubham_test, cv2.COLOR_BGR2RGB)
shubham_test = resize(shubham_test, 0.50)

# finding face location
facelocation_shubham = face_rec.face_locations(shubham)[0]
encode_shubham = face_rec.face_encodings(shubham)[0]
cv2.rectangle(shubham, (facelocation_shubham[3], facelocation_shubham[0]), (facelocation_shubham[1], facelocation_shubham[2]), (255,255,255), 3)


facelocation_shubhamtest = face_rec.face_locations(shubham_test)[0]
encode_shubhamtest = face_rec.face_encodings(shubham_test)[0]
cv2.rectangle(shubham_test, (facelocation_shubham[3], facelocation_shubham[0]), (facelocation_shubham[1], facelocation_shubham[2]), (255,255,255), 3)

results = face_rec.compare_faces([encode_shubham], encode_shubhamtest)
print(results)
cv2.putText(shubham_test, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 2)


cv2.imshow('main_img', shubham)
cv2.imshow('test_img', shubham_test)
cv2.waitKey(0)
cv2.destroyAllWindows()

