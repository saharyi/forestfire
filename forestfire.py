

import pandas as pd
import numpy as np
import  tensorflow as tf
import matplotlib.pyplot as plt


df=pd.read_csv('/home/sahar/Desktop/forestfires.csv')
df.head()

df.describe(include='all')

df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
df.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace =True)

print(df)

print('\n')



n = df.shape[0] 
p = df.shape[1] 

print("n is: ",n)
print('\n')
print("p is: ",p)

print('\n')

df = df.values
train_start = 0 
train_end = int(np.floor(0.5*n))
test_start = train_end           
test_end = n                     
data_train = df[np.arange(train_start, train_end), :]
data_test = df[np.arange(test_start, test_end), :]  


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)


X_train = data_train[:, 0:12] 
y_train = data_train[:, 12] 
X_test = data_test[:, 0:12] 
y_test = data_test[:, 12]  

n_stocks = X_train.shape[1]
print("Number of train_data coulmn without target : ",n_stocks,'\n') 
#inke chanta bezarim va harlaye chanta node dashte bashe residim
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128


net = tf.InteractiveSession()


X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])



sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)


bias_initializer = tf.zeros_initializer()


W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))        
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))          
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))           
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))          


W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))             
bias_out = tf.Variable(bias_initializer([1]))



hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1)) 
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))



out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))




mse = tf.reduce_mean(tf.squared_difference(out, Y))



opt = tf.train.AdamOptimizer().minimize(mse)


net.run(tf.global_variables_initializer())




batch_size =100

mse_train = []
mse_test = []

epochs =2
for e in range(epochs):

    index = np.random.permutation(np.arange(len(y_train))) 
    X_train = X_train[index]                             
    y_train = y_train[index]
    tres = 0.5
    c1=0
    c2=0
   
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
        pred = net.run(out, feed_dict={X: X_test})
        line2.set_ydata(pred)
        mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
        mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
        print('MSE Train: ', (mse_train[-1])*100)
        print('MSE Test: ', (mse_test[-1])*100)

        from sklearn.metrics import accuracy_score
        accuracy_score(y_true=y_test, y_pred=pred, normalize=False)
        print(accuracy_score(y_true=y_test,y_pred =pred,normalize=False))


