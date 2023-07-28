import cv2
import torch


#------------CARGAR MODELO---------
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


#---------CAPTURAR Y PROCESAR IMAGEN O VIDEO-------
def detector():
    cap = cv2.VideoCapture("prueba.jpg") #Captura frame de Imagen, Video o Camara Ip
    
# CICLO WHILE PARA LEER CADA FRAME DEL VIDEO O CAMARA EN VIVO
    while cap.isOpened(): #isOpened valida si hay contenido 
        status, frame = cap.read() #Iterador, que genera los frame. Status dice si se pudo leer o no
        
        if not status:
            break
        
        #INFERENCIA
        pred = model(frame)
        #xmin, ymin, xmax, ymax
        df = pred.pandas().xyxy[0] 

        #Filtrar por confianza
        df = df[df["confidence"]>0.5]

        #-----------DIBUJANDO LAS CAJAS----------
        #Si se usa OpenCV para dibujar las cajas, las coordenadas deben ser de tipo int y no float

        for i in range(df.shape[0]): #Itero sobre cada objeto detectado
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)#Tomo el primer objeto y transformo sus coordenadas en enteros
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(255,0,0), 2) #Dibujo cuadrado(imagen donde va el cuadrado, pto inferior, pto superior, color raya, ancho linea)

            cv2.putText(frame,
                        f"{df.loc[i]['name']}: {round(df.iloc[i]['confidence'],4)}",# Imprimo nombre y confidencia
                        (bbox[0], bbox[1] - 15), # Digo en que posicion va
                        cv2.FONT_HERSHEY_PLAIN, #Que fuente uso
                        1, # grueso
                        (255,255,255),#color
                        2 #tamaño
                        )
            

        cv2.imshow("frame", frame) #Muestra los frame por pantalla, osea reproduce el video. "frame" es el titulo y frame es lo que reproduce

        cv2.waitKey(0)#Espera a que presione una tecla para salir
        #if cv2.waitKey(10) & 0xFF == ord("q"):
         #   break

        cap.release # Se utiliza para liberar los recursos utilizados en la coneccion con la imagen o video






if __name__ == "__main__":
    detector()