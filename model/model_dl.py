from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

def model_predict(input_shape):
    
    # Define the model
    model_seq = Sequential()
    
    #Layer 0
    model_seq.add(Dense(64, activation='relu',input_shape =(43,)))

    #Layer 1
    model_seq.add(Dense(128, activation='tanh'))

    #Layer 2
    model_seq.add(Dense(1, activation='sigmoid'))
    
    #Summary of model architecture
    print (model_seq.summary())
    
    return model_seq