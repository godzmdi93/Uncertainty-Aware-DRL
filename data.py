from keras.datasets import mnist
import pickle

(X_train,y_train),(X_test,y_test) = mnist.load_data()

pickle.dump(X_train,open("xtrain","wb"))
pickle.dump(y_train,open("ytrain","wb"))
pickle.dump(X_test,open("xtest","wb"))
pickle.dump(y_test,open("ytest","wb"))

print('Done')
