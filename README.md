# Face Detection using Haar Cascades with OpenCV and Matplotlib
## Name : Ashwath M
## Register number: 212223230023

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:
```
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('image_01.png',0)
img2 = cv2.imread('image_02.png',0)
img3 = cv2.imread('image_03.png',0)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(img2,cmap='gray');plt.title("With Glass")
plt.subplot(132);plt.imshow(img1,cmap='gray');plt.title("Without sunglass")
plt.subplot(133);plt.imshow(img3,cmap='gray');plt.title("Group Photo")
plt.show()

img1_resized = cv2.resize(img1,(1000,1000))
img2_resized = cv2.resize(img2,(1000,1000))
img3_resized = cv2.resize(img3,(1000,1000))

plt.figure(figsize=[20,20])
plt.subplot(132);plt.imshow(img2_resized,cmap='gray');plt.title("With Glass")
plt.subplot(133);plt.imshow(img3_resized,cmap='gray');plt.title("Group Photo")
plt.subplot(131);plt.imshow(img1_resized,cmap='gray');plt.title("Without sunglass")
plt.show()

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def detect_face(img):

    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.1, minNeighbors=5) # used to find out the location for face

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (127,0,255), 10)
    return face_img


img1_result = detect_face(img1_resized)
img2_result = detect_face(img2_resized)
img3_result = detect_face(img3_resized)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(img1_result,cmap='gray');plt.title('Person(Without Glass)');
plt.subplot(132);plt.imshow(img2_result,cmap='gray');plt.title('Person(With Glass)');
plt.subplot(133);plt.imshow(img3_result,cmap='gray');plt.title('Group of People');
plt.show()

img1 = cv2.imread('image_01.png')
img2 = cv2.imread('image_02.png')
img3 = cv2.imread('image_03.png')

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eye(img):

    face_img = img.copy()

    face_rects = eye_cascade.detectMultiScale(face_img, scaleFactor=1.1, minNeighbors=5) # used to find out the location for face

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,255,0), 2)
    return face_img

img1_resized = cv2.resize(img1,(600,600))
img2_resized = cv2.resize(img2,(600,600))
img3_resized = cv2.resize(img3,(600,600))

img1_result = detect_eye(img1_resized)
img2_result = detect_eye(img2_resized)
img3_result = detect_eye(img3_resized)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(img1_result);plt.title('Person(Without Glass)');
plt.subplot(132);plt.imshow(img2_result);plt.title('Person(With Glass)');
plt.subplot(133);plt.imshow(img3_result);plt.title('Group of People');

img_resized1 = cv2.resize(img3,(1000,1000))
img_resized2 = cv2.resize(img3,(2000,2000))
result1 = detect_eye(img3)
result2 = detect_eye(img_resized1)
result3 = detect_eye(img_resized2)
plt.figure(figsize=(20,40))
plt.subplot(131);plt.imshow(result1);plt.title('Original size(2560x1579)');
plt.subplot(132);plt.imshow(result2);plt.title('Resize of 1000x1000');
plt.subplot(133);plt.imshow(result3);plt.title('Resize of 2000x2000');
plt.show()

cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read()
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Video Face Detection")

while plt.fignum_exists(fig.number):    
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_face(frame)
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)

cap.release()
plt.close()
```

## Output:
<img width="1234" height="522" alt="image" src="https://github.com/user-attachments/assets/eefeee9f-9c25-4cf3-bd2e-9e781079cd41" />

<img width="1222" height="417" alt="image" src="https://github.com/user-attachments/assets/6ecb098b-e607-4328-afa1-092b01b3d564" />
<img width="1233" height="421" alt="image" src="https://github.com/user-attachments/assets/e16b55fd-fb6a-4744-9b79-70b1bc53e3a5" />
<img width="1228" height="422" alt="image" src="https://github.com/user-attachments/assets/a9deb354-09d2-46e0-8bd6-1441d511335d" />
<img width="1230" height="405" alt="image" src="https://github.com/user-attachments/assets/741e437d-2fac-421d-9312-124ed2638761" />
<img width="658" height="503" alt="image" src="https://github.com/user-attachments/assets/c6e5e6c6-103f-4213-a94b-8aa86e640350" />

## Result:
Thus to implement and extend a basic Face Detection System using OpenCV has been executed sucessfully.
