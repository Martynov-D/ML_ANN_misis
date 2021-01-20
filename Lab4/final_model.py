from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from sklearn import metrics
import numpy as np


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def load_cifar100():
    # load dataset
    (trainX10, trainY10), (testX10, testY10) = cifar10.load_data()

    (trainX100, trainY100), (testX100, testY100) = cifar100.load_data(label_mode='coarse')
    (trainX100, trainY100f), (testX100, testY100f) = cifar100.load_data(label_mode='fine')
    trainIndecies = []
    testIndecies = []
    for i in range(len(trainY100)):
        if trainY100[i] != 7:
            trainIndecies.append(i)
    for i in range(len(testY100)):
        if testY100[i] != 7:
            testIndecies.append(i)

    # delete other super classes data
    trainY100f = np.delete(trainY100f, trainIndecies, 0)
    trainX100 = np.delete(trainX100, trainIndecies, 0)
    testY100f = np.delete(testY100f, testIndecies, 0)
    testX100 = np.delete(testX100, testIndecies, 0)

    trainY100f = np.where(trainY100f == 6, 10, trainY100f)
    trainY100f = np.where(trainY100f == 7, 11, trainY100f)
    trainY100f = np.where(trainY100f == 14, 12, trainY100f)
    trainY100f = np.where(trainY100f == 18, 13, trainY100f)
    trainY100f = np.where(trainY100f == 24, 14, trainY100f)

    testY100f = np.where(testY100f == 6, 10, testY100f)
    testY100f = np.where(testY100f == 7, 11, testY100f)
    testY100f = np.where(testY100f == 14, 12, testY100f)
    testY100f = np.where(testY100f == 18, 13, testY100f)
    testY100f = np.where(testY100f == 24, 14, testY100f)

    trainX = np.concatenate((trainX10, trainX100), axis=0)
    trainY = np.concatenate((trainY10, trainY100f), axis=0)
    testX = np.concatenate((testX10, testX100), axis=0)
    testY = np.concatenate((testY10, testY100f), axis=0)

    # one hot encode data
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


# scale pixels
def normalize_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(15, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    pyplot.savefig('diagnostics_plot.png')
    pyplot.close()


# run the test for evaluating a model
def run_test():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = normalize_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    history = model.fit(trainX, trainY, epochs=2, batch_size=64, verbose=0)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    # save model
    model.save('final_model.h5')

def predict(testX):
    model = load_model('final_model.h5')
    result = model.predicct(testX)
    result = np.argmax(result, axis=-1)
    return result

# entry point, run the test
run_test()

trainX, trainY, testX, testY = load_cifar100()

# entry point, run prediction
result = predict(testX)
result_train = predict(trainY)

print(metrics.precision_score(result, testY, average='macro'))
print(metrics.recall_score(result, testY, average='macro'))
print(metrics.f1_score(result, testY, average='macro'))
print(metrics.confusion_matrix(result, testY))

print(metrics.classification_report(testY, result, digits=3))
print(metrics.classification_report(trainY, result_train, digits=3))