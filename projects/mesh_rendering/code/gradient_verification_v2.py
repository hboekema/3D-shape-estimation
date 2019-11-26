import keras
from keras.layers import Input,Embedding,Lambda,Reshape,Concatenate,Flatten,Dense
from keras.optimizers import Adam,SGD
from keras.models import Model
import numpy as np
import keras.backend as K
import os

def learned_gradient(inp1,inp2):
	combined_input = Concatenate()([inp1,inp2])
	print('combined inpuit shape '+str(combined_input.shape))
        #combined_input = Flatten()(combined_input)

        x = Dense(200,activation='relu')(combined_input)
	x = Dense(2,activation='linear')(x) #combined_input)

	#x = Dense(2,activation='linear')(combined_input)
	return x
def network():
	x=Input(shape =(1,))
	x_target = Input(shape =(2,))

	emb=Embedding(C, 2, name="parameter_embedding")(x)
	emb = Reshape((2,))(emb)

	delta = Lambda(lambda x: -x[0]+x[1])([x_target,emb]) #difference between predicted value and target value, note that sign is important
	true_grad = Lambda(lambda x: 2*x[0]-2*x[1])([emb,x_target])

	update = learned_gradient(emb,x_target)	
	
	#diff =  Lambda(lambda x:K.square(x[0]-x[1]))([true_grad,update])  #learning to approximate local gradient
	#diff =  Lambda(lambda x:K.square(x[0]-x[1]))([x_target,update])   #learning to approximate target values
	diff =  Lambda(lambda x:K.square(x[0]-x[1]))([delta,update]) #learning to approximate difference between predicted value and target - like in our SMPL example

	print('update shape '+str(update.shape))
	update_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(update)
	mult_output = Lambda(lambda x: x[0]*x[1])([emb,update_NOGRAD])

	return [x,x_target],[emb,update_NOGRAD,mult_output,diff]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
N=100000
C=3
x_train = [np.zeros((N,1))+np.reshape(np.arange(N)%C,(N,1)),np.zeros((N,2))+np.random.rand(N,2)]
#print('x train -1 '+str(x_train[-1]))
#exit(1)
y_train = [np.zeros((N,2)),np.zeros((N,2)),np.zeros((N,2)),np.zeros((N,2))]

N_test=1
x_test =  [np.zeros((N_test,1))+np.reshape(np.arange(N_test)%C,(N_test,1)),np.zeros((N_test,2))+np.random.rand(N,2)]
y_test = [np.zeros((N_test,2)),np.zeros((N_test,2)),np.zeros((N_test,2)),np.zeros((N_test,2))]

inputs, outputs = network()
model =Model(inputs=inputs,outputs=outputs)

#y_pred= model.predict(x_train)
#print('y_pred '+str(y_pred))
#print('y_train ' +str(y_train))

learning_rate = 0.01
optimizer= Adam(lr=learning_rate, decay=0.000)
#optimizer= SGD(lr=learning_rate,momentum=0.0, nesterov=False)

def false_loss(y_true,y_pred):
	return K.sum(y_pred)
def no_loss(y_true,y_pred):
	return K.sum(y_true*0)

model.compile(optimizer=optimizer, loss=[no_loss,no_loss, false_loss,false_loss], loss_weights=[0.0,0.0,1.0,1.0])

for i in range(1):
	model.fit(x=x_train,y=y_train,batch_size=100,epochs=10)

y_pred = model.predict(x_train)
print('y_pred '+str(y_pred))
#exit(1)



#
emb_layer=model.get_layer('parameter_embedding')
emb_weights=emb_layer.get_weights()
emb_weights[0]= emb_weights[0]+0.1

emb_layer.set_weights(emb_weights)
print('Starting weights '+str(emb_layer.get_weights()))

print('x test [1] [0,:]' +str(x_test[1][0,:]))

for i in range(50):
	#print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))	
	y_pred = model.predict(x_test)
	print('y_pred [0] [0:,] '+str(y_pred[0][0,:]))
	emb_weights=emb_layer.get_weights()
	#emb_weights[0][0,:]=0*emb_weights[0][0,:]+y_pred[1][0,:]
	emb_weights[0][0,:]=emb_weights[0][0,:]-y_pred[1][0,:]*0.5

	emb_layer.set_weights ( emb_weights)



print('x test [1] [0,:]' +str(x_test[1][0,:]))
