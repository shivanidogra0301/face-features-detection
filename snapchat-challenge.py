import cv2

glasses = cv2.imread("./glasses.png", -1)
moustache = cv2.imread("./mustache.png", -1)
image = cv2.imread("./Jamie_Before.jpg")
image = cv2.resize(image, (500, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

eye_classifier = cv2.CascadeClassifier("./frontalEyes35x16.xml")
nose_classifier = cv2.CascadeClassifier("./Nose18x15.xml")

eyes = eye_classifier.detectMultiScale(image_gray, 1.5, 5)
for (x, y, w, h) in eyes:
    # cv2.rectangle(image,(x,y),(x+h,y+w),2)
    glasses = cv2.resize(glasses, (w, h))

    gw, gh, gc = glasses.shape
    for i in range(0, gw):
        for j in range(0, gh):
            if glasses[i, j][3] != 0:
                image[y + i, x + j] = glasses[i, j]

nose = nose_classifier.detectMultiScale(image_gray, 1.5, 5)
for (x, y, w, h) in nose:
    moustache = cv2.resize(moustache, (w, h))

    mw, mh, mc = moustache.shape
    for i in range(mw):
        for j in range(mh):
            if moustache[i, j][3] != 0:
                image[y + i + int(h / 2.0), x + j] = moustache[i, j]

cv2.imshow("Jamie", image)
# cv2.imshow("glasses",glasses1)

cv2.waitKey(0)
cv2.destroyAllWindows()