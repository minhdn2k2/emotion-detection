import cv2
import numpy as np

idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 
                          4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

def draw_face(img, faces, y_pred, color=(0,255,0)):
    # y_pred là tất cả những cảm xúc mà recognizer đã dự đoán
    # faces là tất cả những điểm vị trí mà detector dự đoán (tương ứng với y_pred)

    # Draw box and write emotion text to each face
    # Lặp qua từng điểm vị trí của từng khuôn mặt và sau đó vẽ khung và tên cảm xúc
    for index, face in enumerate(faces):

        bbox = face['bbox']  # Điểm vị trí của 1 khuôn mặt

        # vẽ khung lên 1 khuôn mặt
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])), # vị trí của khung
            (int(bbox[2]), int(bbox[3])), # vị trí của khung
            color, # màu của khung
            2 # độ dày khung
        )

        # viết chữ emotion lên 1 khuôn mặt
        cv2.putText(
            img,
            idx_to_class[y_pred[index]], # nội dung của chữ
            (int(bbox[0]) + 0, int(bbox[1]) - 10), # vị trí của chữ
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, # kích thước chữ
            color, # màu của chữ
            2,  # kích thước chữ
            cv2.LINE_AA
        )

    # Viết thống kê các cảm xúc có trong bức ảnh
    # lặp qua từng cảm súc
    # for i in range(len(idx_to_class)):
    #     cv2.putText(
    #         bbox_array,
    #         f'{idx_to_class[i]} : {list(y_pred).count(i)}', # nội dung của chữ
    #         (20, (i + 1) * 30), # vị trí chữ
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1, # kích thước chữ
    #         colors[i], # màu của chữ
    #         2, # kích thước chữ
    #         cv2.LINE_AA
    #     )