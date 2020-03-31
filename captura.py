import cv2
import numpy as np

classificateFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificateEye = cv2.CascadeClassifier("haarcascade_eye.xml")

camera = cv2.VideoCapture(1)
amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print("Capturando as faces...")

while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces = classificateFace.detectMultiScale(
        imagemCinza,
        scaleFactor=1.1,
        minNeighbors=5,  
        minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 0, 255), 2)
        eye = imagem[y:y + h, x:x + w]
        grayEye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        detectedEyes = classificateEye.detectMultiScale(grayEye)
        for(ex, ey, ew, eh) in detectedEyes:
            cv2.rectangle(eye, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 110:
                    imagemFace = cv2.resize(imagemCinza [y:y + h, x:x + w], (largura, altura))
                    cv2.imwrite("fotos/pessoa."+str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1
                else:
                    print("Não há iluminação suficiente")

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso")
camera.release()
cv2.destroyAllWindows()
