from tensorflow.keras.models import load_model
import numpy as np

model = load_model('emotion-detection/mobilenet_7.h5')



def predict(X_pred):
    Y_pred = []
    for i in range(len(X_pred)):
        face = X_pred[i].reshape(1,224,224,3)
        y_pred = model.predict(face)
        i = np.argmax(y_pred)
        Y_pred.append(i)
    return Y_pred
