from insightface.app import FaceAnalysis
from my_utils.dataset import transfer_data, transfer_data_by_crop
from my_utils.model import predict
from my_utils.draw import draw_face
import cv2
import os
import numpy as np
import time
idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 
                          4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

def run_camera():
    detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
                            allowed_modules=['detection'], name='buffalo_sc')
    detector.prepare(ctx_id=0, det_size=(640, 640))
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        # detector
        faces = detector.get(img)
        # get faces
        X_pred = transfer_data_by_crop(faces, img)
        # predict emotion
        y_pred = predict(X_pred)
        draw_face(img, faces, y_pred)
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
    
    
    
    
    