from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

def model_predict(input_shape):
    
    # Define the model
    model_seq = Sequential()
    
    #Layer 0
    model_seq.add(Dense(64, activation='tanh',input_shape =(43,)))

    #Layer 1
    model_seq.add(Dense(32, activation='relu'))

    #Layer 2
    model_seq.add(Dense(1, activation='sigmoid'))
    
    #Summary of model architecture
    print (model_seq.summary())
    
    return model_seq