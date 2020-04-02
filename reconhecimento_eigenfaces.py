import cv2

faceDetector = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("classificadores/classificadorEigen.yml")
width, height = 220, 220

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(1)

while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    detectedFace = faceDetector.detectMultiScale(
        imagemCinza,
        scaleFactor= 1.5,
        minNeighbors=5,
        minSize=(30, 30))

    for (x, y, w, h)in detectedFace:
        imagemFace = cv2.resize(imagemCinza [y:y + h, x:x + w ], (width, height))
        cv2.rectangle(imagem, (x,y), (x + w, y + h), (0, 0, 255), 2)
        id, confidence = recognizer.predict(imagemFace)
        cv2.putText(imagem, str(id), (x,y + (h+30)), font, 2, (0, 0, 255))
        cv2.putText(imagem, str(confidence), (x, y + (h+50)), font, 1, (0,0,255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()