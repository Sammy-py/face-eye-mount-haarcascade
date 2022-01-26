import cv2

yuz_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
oz_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
agiz_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#video = cv2.imread("bb.jpeg")
video = cv2.imread('D:/ARGE/team/devam/denenen resimler/a.jpg') 

esikleme = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
yuz = yuz_cascade.detectMultiScale(esikleme, 1.1, 20)           # ... = (eşiklenen resim, scaleFactor, minNeighbors)

for (x, y , w ,h) in yuz:
    cv2.rectangle(video, (x,y), (x+w, y+h), (255, 0 , 0), 2)
    roi_esikleme = esikleme[y:y+h, x:x+w]
    roi_renk = video[y:y+h, x:x+w]
    goz = goz_cascade.detectMultiScale(roi_esikleme, 1.2 ,12)     # ... = (eşiklenen resim, scaleFactor, minNeighbors)
    agiz = agiz_cascade.detectMultiScale(roi_esikleme, 1.2, 22)  # ... = (eşiklenen resim, scaleFactor, minNeighbors)
        
    for (ex, ey ,ew, eh) in goz:
        cv2.rectangle(roi_renk, (ex,ey), (ex+ew, ey+eh), (0, 0, 0), -1) #-1 demek tespit edilen bçlgenin tamamını kaplamak demek
            
    for (sx, sy ,sw, sh) in agiz:
         cv2.rectangle(roi_renk, (sx,sy), (sx+sw, sy+sh), (0, 0, 0), -1)


cv2.imshow('sonuc', video)

cv2.imwrite("sonuc.png", video) # video isimli resmi bulunduğu dosyaya kayıt eder. png, jpg, jpeg .....

cv2.waitKey(0)
cv2.destroyAllWindows()
