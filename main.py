from imutils import face_utils

import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import sklearn
import imutils
import pickle
import math
import dlib
import cv2

# Carregue o detector de faces do dlib e o preditor de landmarks

selected_model = 'SVM'

if selected_model == 'XGBoost':
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, random_state=42)
    model.load_model('models/xgb_model.json')
elif selected_model == 'SVM':
    model = pickle.load(open('models/svm_model.sav', 'rb'))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
tempo_de_sonolencia = 0

def calcular_distancia(x1, y1, x2, y2):
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia

def get_distance_test(pt1, pt2, shape):
  x1,y1 = shape[pt1 - 1]
  x2,y2 = shape[pt2 - 1]
  return calcular_distancia(x1,y1,x2,y2)

# ESP32 URL
URL = "http://192.168.3.109"

cap = cv2.VideoCapture(URL + ":81/stream")
fps = int(cap.get(cv2.CAP_PROP_FPS))

if __name__ == '__main__':

    while True:

        if cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            # Process the 'frame' here, for example, display it
            image = imutils.resize(frame, height=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w, _ = image.shape

            # detect faces in the grayscale image
            rects = detector(gray, 1)

            if not rects:
                requests.get(URL + f'/control?var=quality&val={10}"')
            # loop over the face detections
            for (i, rect) in enumerate(rects):
                
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                ref = get_distance_test(37, 40, shape)
                distancia = [
                    get_distance_test(38, 42, shape)/ref,
                    get_distance_test(39,41, shape)/ref,
                    get_distance_test(44,48, shape)/ref,
                    get_distance_test(45,47, shape)/ref,
                    get_distance_test(63,67, shape)/ref,
                    get_distance_test(62,68, shape)/ref,
                    get_distance_test(64,66, shape)/ref
                    ]

                data = pd.DataFrame([distancia], columns=["0", "1", "2", "3", "4", "5", "6"])
                previsao = model.predict(data)
                qualidade = previsao[0] + 10

                #Retorna o dado para o microcontrolador
                requests.get(URL + f'/control?var=quality&val={qualidade}"')
                if previsao[0] == 0:
                    texto = 'Alerta'
                    tempo_de_sonolencia = 0
                elif previsao[0] == 1:
                    tempo_de_sonolencia += (1 /fps)
                    texto = 'Sonolento'
                else:
                    requests.get(URL + "/action?go=forward")
                    texto = 'Bocejando'
                    tempo_de_sonolencia += (1 /fps)


                texto2 = f'Tempo olho Fechado: {tempo_de_sonolencia:.2f}s'
                fonte = cv2.FONT_HERSHEY_SIMPLEX
                tamanho_fonte = 0.5
                cor = (255, 255, 255)  # Cor do texto (no formato BGR)
                espessura = 2
                largura_texto, altura_texto = cv2.getTextSize(texto, fonte, tamanho_fonte, espessura)[0]
                altura_imagem, largura_imagem, _ = image.shape
                posicao_x = largura_imagem - largura_texto - 10  # 10 pixels de margem à direita
                posicao_y = altura_texto + 10  # 10 pixels de margem acima

                # Coloca os resultados na imagem
                cv2.putText(image, texto, (posicao_x, posicao_y), fonte, tamanho_fonte, (0,0,0), 4)
                cv2.putText(image, texto2, (posicao_x - 220, posicao_y + 30), fonte, tamanho_fonte, (0,0,0), 4)
                cv2.putText(image, texto, (posicao_x, posicao_y), fonte, tamanho_fonte, cor, espessura)
                cv2.putText(image, texto2, (posicao_x - 220, posicao_y + 30), fonte, tamanho_fonte, cor, espessura)

            cv2.imshow("frame", image)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()