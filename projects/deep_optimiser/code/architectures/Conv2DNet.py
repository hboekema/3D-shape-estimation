

from keras.layers import Lambda, Conv2D, Reshape, Dropout, Dense

def Conv2DNet(input_layer, DROPOUT, TRAINABLE):
        #assert DROPOUT == 0.0
        print('input_layer shape: '+str(input_layer.shape))

        input_layer = Dropout(DROPOUT)(input_layer)
        optlearner_architecture = Conv2D(64, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(input_layer)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(128, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(256, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(512, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        #exit(1)

        optlearner_architecture = Reshape((-1,))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Dense(31*3, activation="linear", trainable=TRAINABLE)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Reshape((31, 3))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))

        return optlearner_architecture


def Conv2DNet_deeper(input_layer, DROPOUT, TRAINABLE):
        #assert DROPOUT == 0.0
        print('input_layer shape: '+str(input_layer.shape))

        input_layer = Dropout(DROPOUT)(input_layer)
        optlearner_architecture = Conv2D(64, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(input_layer)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(128, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(256, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(512, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(1024, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv2D(2048, (3,3), strides=2, activation="relu", trainable=TRAINABLE)(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        #exit(1)

        optlearner_architecture = Reshape((-1,))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Dense(31*3, activation="linear", trainable=TRAINABLE)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Reshape((31, 3))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))

        return optlearner_architecture


