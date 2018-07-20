from sys import argv, exit
import argparse
import json
from pprint import PrettyPrinter

from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.utils import to_categorical

from numpy import std, array, append, zeros, shape, mean, squeeze 
from scipy.misc import imread, imresize

"""
This class contains all the methods required to load the input dataset, preprocess data,
train and test both the benchmark model and the CNN. 
"""
class LanguageTrainer:

    """
    The method execution flow is executed from the constructor itself
    """
    def __init__(self):
        self.args = self.parse_args()
        with open(self.args.classes) as classes_json:
            self.classes = json.load(classes_json)
        self.model_load_from = self.args.model
        self.saveas = self.args.saveas
        self.epochs = self.args.epochs
        self.get_images()
        if self.args.resize:
            self.set_resize_shape()
        self.encode_labels()
        self.load_model()
        if self.args.predict:
            self.predict()
        else:
            self.train_model()


    """
    From the classes.json file passed, load all the files from the paths given in the keys and put it in the images list. 
    """
    def get_images(self):
        self.images = []
        for folder_name in self.classes:
            self.images += [join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]


    """
    Prints the error message and exits the program.  
    """
    def die(self, err):
        exit("\nError: {}, exiting".format(err))


    """
    Validates arguments passed
    """
    def validate_args(self, args):
        # It is mandatory to pass classes json. If not found, the program will exit
        if not args.classes:
            self.die("classes.json is required")
        try:
            open(args.classes)
        except IOError:
            self.die("classes.json not found")


    """
    Data preprocessing step: if the images are not of the same shape, they'll be reshaped to 
    (mean of number of rows, mean of number of cols, 3). 3 is because the images are being read
    in RGB mode.
    """
    def set_resize_shape(self):
        # If the shape is provided, us as it is. 
        if self.args.shape:
            sh = self.args.shape.split(",")
            self.rows = int(sh[0])
            self.cols = int(sh[1])
        else:    
            rows, cols = 0, 0
            for img in self.images:
                imgMatrix = imread(img, mode='RGB')
                if imgMatrix is None:
                    continue
                rows += imgMatrix.shape[0]
                cols += imgMatrix.shape[1]
            self.rows = rows // len(self.images)
            self.cols = cols // len(self.images)
            print "Resizing images to shape [{},{}]".format(self.rows, self.cols)


    """
    All the available options while running the program
    """
    def parse_args(self):
        parser = argparse.ArgumentParser()
        required_args = parser.add_argument_group("Required Arguments")
        required_args.add_argument("-classes", help="json containing folder path as the key and class as value")
        parser.add_argument("-model", help="pass an already trained model for further training")
        parser.add_argument("-shape", help="shape the images should be resized to")
        parser.add_argument("-saveas", default="output_model.h5", help="save the model as")
        parser.add_argument("-epochs", default=200, type=int)
        parser.add_argument("--no-resize", dest="resize", action="store_false", help="all images are of same shape, no need to resize")
        parser.add_argument("-resize", dest="resize", action="store_true", help="all images are not of same shape, resize")
        parser.add_argument("-pred", dest="predict", action="store_true", help="run prediction instead of training")
        parser.add_argument("--fully-connected-only", dest="fc", action="store_true", help="Use only fully connected layer")
        parser.set_defaults(predict=False)
        parser.set_defaults(fc=False)
        parser.set_defaults(download=False)
        parser.set_defaults(resize=True)
        args = parser.parse_args()
        self.validate_args(args)
        return args


    """
    This program is equipped to support multi-label, multi-class classification
    """
    def encode_labels(self):
        from sklearn.preprocessing import MultiLabelBinarizer 
        labels = []
        # labels in string form are encoded using MultiLabelBinarizer
        for k, v in self.classes.iteritems():
            labels.append([l.strip() for l in v.split(",")])
        self.label_encoder = MultiLabelBinarizer()
        print "Transforms: {}".format(self.label_encoder.fit_transform(labels))        
        print "Output classes: {}".format(self.label_encoder.classes_)
 

    """
    This method is responsible for supplying the input to training and testing methods. 
    """
    def get_dataset(self, start, end):
        x,y = [],[]
        for i in range(start, end):
            img = self.images[i]
            # Read the image as a matrix in RGB mode
            imgMatrix = imread(img, mode='RGB')
            if imgMatrix is None:
               continue
            if self.args.resize:
                # Resize the images to a common shape
                imgMatrix = imresize(imgMatrix, (self.rows, self.cols, 3)) 
                if self.args.fc:
                    # If using only fully connected layers, transform the matrix into a vector
                    imgMatrix = array(imgMatrix).flatten()
            # Append the image matrix to the list of input matrices
            x.append(imgMatrix)
            labels = str(self.classes[img[:img.rfind("/")]]).split(",")
            # Append the labels to output list
            y.append(squeeze(self.label_encoder.transform([labels])))
        
        # Normalize by subtracting the mean and dividing by the standard deviation
        x -= mean(x)
        x /= std(x)
        return array(x), array(y)


    """
    This is where the model configuration is decided based on the parameters passed
    """
    def load_model(self):
        # If a model has been passed, use it.
        if self.model_load_from:
            self.model = keras.models.load_model(self.model_load_from)
        else:
            if self.args.predict:
                # If the user intended to predict classes and forgot to pass a model, die
                die("Pass a model for prediction")
            elif self.args.fc:
                # If --fully-connected-only flag is set, build a Fully Connected Neural Network.
                self.model = Sequential()
                self.model.add(Dense(32, activation='relu', input_shape=(self.rows * self.cols * 3, )))
                self.model.add(Dense(64, activation='relu'))
                self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))
            else:                
                # Otherwise build a CNN
                self.model = Sequential()
                
                # First set of layers    
                self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
                self.model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
                self.model.add(Dropout(0.25))

                # Second set of layers
                self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
                self.model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
                self.model.add(Dropout(0.25))
    
                # Third set of layers
                self.model.add(Conv2D(64, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
                self.model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
                self.model.add(Dropout(0.25))

                # Fourth set of layers
                self.model.add(Conv2D(128, (3, 3), activation='relu')) 
                self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                self.model.add(Dropout(0.25))

                # Flattening the input to be passed onto Fully Connected Layers
                self.model.add(Flatten())

                # First fully connected layer
                self.model.add(Dense(128, activation='relu'))
                self.model.add(Dropout(0.5))

                # Last layer, responsible for predicting the output
                self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

    
    """
    This function is called whenever -pred flag is passed. Instead of training the model, the passed model will 
    be used to predict labels for the images present in the paths provided in classes json. 
    """
    def predict(self):
        x_test, y_test = self.get_dataset(0, len(self.images))
        res = self.model.predict(x_test, batch_size=128, verbose=1)    
        p = PrettyPrinter(indent=4) 
        
        print "\nEvaluation on test data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(x_test, y_test, batch_size=128))))


    """
    This method is responsible for training the model. 
    """
    def train_model(self):
        # Chosen loss function is 'categorical_crossentropy' and chosen optimizer is 'adam'.
        # Accuracy will be used to observe model's performance. 
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Create checkpoint after every epoch
        cb = [keras.callbacks.ModelCheckpoint(self.saveas[:-3] + "_cp.h5", monitor='acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)]
        
        x_train, y_train = self.get_dataset(5, len(self.images))
        x_val, y_val = self.get_dataset(0, 5)
        
        # Fit the model on the training data
        self.model.fit(x_train, y_train,
                  epochs=self.epochs,
                  batch_size=64,
                  callbacks=cb
                )   

        # Save the model with the value passed for -saveas argument
        self.model.save(self.saveas)

        print "\nEvaluation on validation data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(x_val, y_val, batch_size=128))))

if __name__=="__main__":
    lt = LanguageTrainer()
