

import hashlib, os, optparse, sys
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
import random
import os
import cv2
import imghdr
import json
from PIL import Image
import tensorflow as tf
from simple_image_download import simple_image_download as simp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.preprocessing import image as KerasImage
import tensorflow_hub as hub
import datetime



# global vars data directories
model_dir = "model"
dataset_dir = "content/datasets"
base_dir = os.path.join(dataset_dir, "photos")


class ArtstyleConfig():
    def __init__(self):
        #Tensorflow 2.0 has some problems. Found this solution online, idk why tho
        self.config_gpu()

         # set image shape to (224, 224) in order to use MobileNet v2 network
        self.IMAGE_SIZE = 224

        # set batch size for training
        self.BATCH_SIZE = 32

        # read model classes definition from json file (needed to decode prediction results)
        self.CLASS_INDEX = self.load_model_class()
        
        print("-------------------------")
        print("IMAGE_SIZE: {}".format(self.IMAGE_SIZE))
        print("BATCH_SIZE: {}".format(self.BATCH_SIZE))
        print("MODEL_CLASSES: {}".format(self.CLASS_INDEX))
        print("-------------------------")
        
    def class_num(self):
        return len(self.CLASS_INDEX)

    def class_names(self):
        return self.CLASS_INDEX.values()

    def decode_classid(self, class_id):
        return self.CLASS_INDEX[class_id]

    def load_model_class(self):
        return json.load(open("model_class.json"))

    def save_model_class(self, train_classes):
        with open('model_class.json', 'w') as f:
            json.dump(train_classes, f, indent=4)

    def config_gpu(self):
        """GPU setting globally. Tensorflow 2.0 reports this CUBLAS error (don't know why though). 
           Found this solution online. May be changed or unnecessary in future versions
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs are initialized
                print(e)


#======================================
# Artstyle dataset class
#======================================
class ArtstyleDataset():
    def __init__(self, config):
        self.config = config

    def prepare(self):
        # download imgs from Google 
        self.download_from_google("cubism", [
            ('cubism picasso', 400),
            ('juan gris cubism', 200),
            ('georges braque cubism', 100),
            ('jean metzinger cubism',200)
        ])

        self.download_from_google("ghibli", [        
            ('hayao miyazaki sketches watercolor', 300),
            ('hayao miyazaki sketches watercolor nausicaa', 200),
            ('hayao miyazaki sketches watercolor kiki''s delivery', 100),
            ('hayao miyazaki sketches watercolor concept art', 300),
            ('hayao miyazaki sketches watercolor castle', 50),
            ('hayao miyazaki sketches watercolor princess mononoke', 50),
            ('hayao miyazaki sketches ponyo', 300),
            ('hayao miyazaki sketches fireflies', 300),
            ('hayao miyazaki sketches watercolor porco rosso', 100),
            ('hayao miyazaki sketches watercolor wind', 200),
        ])

        self.download_from_google("impressionism", [
            ('monet', 500),
            ('corot impressionism', 300),
            ('van gogh original paintings', 400),
            ('monet water lilies', 50),
            ('monet haystacks', 50),
            ('monet trees', 200),
        ])

        # remove duplicated images
        self.remove_duplicated_images()
        self.tile()
        self.split()

    #-------------------------------------
    # Download images from Google Search
    #-------------------------------------
    def download_from_google(self, artstyle, search_list):
        """download images from google search by given keywords
        """
        if not search_list:   # empty list
            return

        # create artstyle directory
        artstyle_dir = os.path.join(dataset_dir, artstyle)
        if not os.path.exists(artstyle_dir):
            os.makedirs(artstyle_dir)

        response = simp.simple_image_download()

        for item in search_list:
            keystr, limit = item

            # downloaded img automatically saved in "simple_images/keywords" folder
            response.download(keystr, limit)

            # move the downloaded images to the artstyle folder     
            download_dir =  os.path.join("simple_images", keystr.replace(' ', '_'))
            images = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
            
            for img in images:
                shutil.move(img, artstyle_dir)


    #-------------------------------------
    # Remove duplicated images
    #-------------------------------------
    def md5(self, f):
        """takes one file f as an argument and calculates 'md5 checksum' for that file
        """
        md5Hash=hashlib.md5()
        with open(f,'rb') as f:
            for chunk in iter(lambda: f.read(4096),b""):
                md5Hash.update(chunk)
        return md5Hash.hexdigest()


    def rm_dup(self, path, exps=""):
        """relies on the md5 function above to remove duplicate files in the given directory
        """
        if not os.path.isdir(path):     #make sure the given directory exists
            print('specified directory does not exist!')
        else:
            md5_dict={}
            if exps:
                exp_list=exps.split("-")
            else:
                exp_list = []
            print("Working in path {} ...\n".format(path))
            
            for root, dirs, files in os.walk(path):    #the os.walk function allows checking subdirectories too
                for f in files:
                    filePath=os.path.join(root,f)
                    md5Hash=self.md5(filePath)
                    size=os.path.getsize(filePath)
                    fileComb=str(md5Hash)+str(size)
                    if fileComb in md5_dict:
                        md5_dict[fileComb].append(filePath)
                    else:
                        md5_dict.update({fileComb:[filePath]})
            ignore_list=[]
            for key in md5_dict:
                for item in md5_dict[key]:
                    for p in exp_list:
                        if item.endswith(p):
                            ignore_list.append(item)
                            while md5_dict[key].count(item)>0:
                                md5_dict[key].remove(item)

            print("Done! The following files will be deleted:\n")
            for key in md5_dict:
                for item in md5_dict[key][:-1]:
                    print(item)
            if input("\nEnter (y)es to confirm operation or anything else to abort: ").lower() not in ("y", "yes"):
                sys.exit("Operation cancelled by user. Exiting...")

            print("Deleting...")
            c=0
            for key in md5_dict:
                while len(md5_dict[key])>1:
                    for item in md5_dict[key]:
                        os.remove(item)
                        md5_dict[key].remove(item)
                        c += 1
            if len(ignore_list)>0:
                print('Done! Found {} duplicate files, deleted {}, and ignored {} on user\'s request...'.format(c+len(ignore_list),c,len(ignore_list)))
            else:
                print('Done! Found and deleted {} files...'.format(c))


    def remove_duplicated_images(self):
        """ remove duplicatd images in all subfolders of the dataset"""

        print("Removing duplicated imgs")
        
        for artstyle_dir in self.config.class_names():
            self.rm_dup(os.path.join(dataset_dir, artstyle_dir))

        # for root, dirs, files in os.walk(path): 
        #     for d in dirs:
        #         self.rm_dup(os.path.join(root, d))

        print("Done")


    # -------------------------------------
    # Tile imgs to increase # of imgs
    # -------------------------------------
    def plotImages(self, images, cols, cvformat=True):
        rows = len(images) // cols + 1
        plt.figure(figsize=(20, 20 * rows // cols))
        i = 1
        for img in images:
            plt.subplot(rows, cols, i)
            
            if cvformat:
                img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_cvt)
            
            plt.imshow(img)
            i += 1

        plt.show()


    def tile_image(self, img_path, plot=True, save2file=False):

        img = cv2.imread(img_path)
 
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        ratio = imgHeight / imgWidth
        
        # try to make tile size close to (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        tile_size = int(self.config.IMAGE_SIZE * 1.2)
        
        if ratio > 1:
            numcols = min(2, int(imgWidth // tile_size) + 1)      # max 2 tiles in a column
            numrows = numcols
            if (ratio > 1.25):
                numrows = numcols + int(ratio)
        else:
            numrows = min(2, int(imgHeight // tile_size) + 1)     # max 2 tiles in a row
            numcols = numrows
            if (ratio < 0.5): 
                numcols = numrows + int(1//ratio)        

        images = [img]
        
        if numrows > 1 or numcols > 1:
            height = int(imgHeight / numrows)
            width = int(imgWidth / numcols)

            for row in range(numrows):
                for col in range(numcols):
                    y0 = row * height
                    y1 = y0 + height
                    x0 = col * width
                    x1 = x0 + width
                    img_tile = img[y0:y1, x0:x1]
                    
                    if save2file:
                        # save  tile to file
                        filename = "{}_tile_{}{}.jpg".format(os.path.splitext(img_path)[0], row, col)
                        
                        cv2.imwrite(filename, img_tile)
            
                    images.append(img_tile)

        # if plot:
        #     self.plotImages(images, cols=10, cvformat=True) 

        return images


    def tile(self):    # the big tile func
        for cl in self.config.class_names():
            class_dir = os.path.join(dataset_dir, cl)
            
            # get all image paths in this directory
            img_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            img_paths = [p for p in img_paths if imghdr.what(p)]   # remove non-image files from the list

            # tile images in this folder
            for img in img_paths:
                self.tile_image(img, plot=False, save2file=True)
                
            print("{}: tiling completed".format(cl))        


    #-------------------------------------
    # Split dataset into training and validation
    #-------------------------------------
    def split(self):
        for cl in self.config.class_names():
            class_dir = os.path.join(dataset_dir, cl)

            # get all images including tiles
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            images = [p for p in images if imghdr.what(p)]   # remove non-image files from the list
            
            # shuffle the images to separate tiles
            random.shuffle(images)
            
            # split train and val
            cut = round(len(images)*0.8)    # train : val = 80 : 20
            train, val = images[:cut], images[cut:]
            
            train_path = os.path.join(base_dir, 'train', cl)
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            for img in train:
                shutil.move(img, train_path)
            print("{} Images from {} => {}".format(len(train), class_dir, train_path))

            val_path = os.path.join(base_dir, 'val', cl)
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            for img in val:
                shutil.move(img, val_path)
            print("{} Images from {} => {}".format(len(val), class_dir, val_path))

        # # set up the path for the training and validation datasets for convenience
        # train_dir = os.path.join(base_dir, 'train')
        # val_dir = os.path.join(base_dir, 'val')

        # print("train_dir: {}".format(train_dir))
        # print("val_dir: {}".format(val_dir))


#======================================
# Art style model class
#======================================
class ArtstyleModel():
    def __init__(self, mode, config, model_path=""):
        assert mode in ['training', 'inference']

        self.mode = mode
        self.config = config

        if mode == "training":
            self.keras_model = self.build()
        else:
            self.keras_model = self.reload(model_path)

        self.keras_model.summary()

    
    def build_CNN(self):
        """ build Convolutional neural network (CNN) layer by layer """

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.config.class_num(), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
       
        return model


    def build(self):
        """ build Convolutional neural network (CNN) by transfer learning from MobileNet v2 """
        
        # Extract features from the pre-trained Mobilenet v2 model
        feature_extractor = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                                           input_shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3))

        # Set the pre-Trained layers "untrainable" so that the parameters in these layers
        # will not be changed during the training of our model
        feature_extractor.trainable = False

        # Create a keras Sequential model, with its initial layers above pre-trained layers,
        # and the last layer a dense layer for art style features
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(self.config.class_num(), activation='softmax')
        ])


        # compile the model with a optimizer and a loss function
        # also set accuracy as the metrics argument to look at during the training
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model


    #-------------------------------------
    #Reload keras model from .h5 file
    #-------------------------------------
    def reload(self, model_path):
        model = tf.keras.models.load_model(
            model_path, 
            # `custom_objects` tells keras how to load a `hub.KerasLayer`
            custom_objects={'KerasLayer': hub.KerasLayer}
        )

        print("\nModel loaded from: {}\n".format(model_path))

        return model


    #-------------------------------------
    # train the model with dataset
    #-------------------------------------
    def train(self, dataset_dir, epochs=30):
        
        # data generator for training
        image_gen_train = ImageDataGenerator(
            rescale=1./255, 
            horizontal_flip=True,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.5,
            fill_mode='nearest'
        )

        train_data_gen = image_gen_train.flow_from_directory(batch_size=self.config.BATCH_SIZE,
                                                             directory=os.path.join(dataset_dir, "train"),
                                                             shuffle=True,
                                                             target_size=(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE),
                                                             class_mode='sparse')


        # data generator for validation
        image_gen_val = ImageDataGenerator(rescale=1./255)
        val_data_gen = image_gen_val.flow_from_directory(batch_size=self.config.BATCH_SIZE,
                                                         directory=os.path.join(dataset_dir, "val"),
                                                         target_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                                                         class_mode='sparse')

        #=========================================
        #Read class list from image generator

        labels = (train_data_gen.class_indices)
        train_classes = dict((v,k) for k,v in labels.items())
        print("Classes for training: {}\n".format(train_classes))

        num_train = len(train_data_gen.filenames)
        num_val = len(val_data_gen.filenames)
        print("Loaded {} images for training".format(num_train))
        print("Loaded {} images for validation".format(num_val))

        # # Create logs directory to save weight models
        # self.log_dir = os.path.join(self.__trained_model_dir, "artstyle_{:%Y%m%dT%H%M}".format(datetime.datetime.now()))
        # if not os.path.isdir(self.log_dir):
        #     os.makedirs(self.log_dir)

        # # callback: only save the best model
        # callbacks = [
        #     keras.callbacks.ModelCheckpoint(
        #         filepath=os.path.join(self.log_dir, 'artstyle_{epoch:03d}-{val_acc:.5f}.h5'),
        #         monitor='val_acc',
        #         verbose=1,
        #         save_best_only=True,
        #         save_weights_only=True,
        #         period=1),
        #     keras.callbacks.LearningRateScheduler(self.lr_schedule)
        # ]

        # train the model
        history = self.keras_model.fit(
            train_data_gen, 
            validation_data=val_data_gen,
            steps_per_epoch=int(num_train / float(self.config.BATCH_SIZE)), 
            validation_steps=int(num_val / float(self.config.BATCH_SIZE)), 
            # callbacks=callbacks,
            epochs=epochs,
        )

        #Save the trained model as .h5
        val_acc = history.history["val_accuracy"]

        model_path = os.path.join(model_dir, "artstyle_{:%Y%m%dT%H%M}_{:.4f}.h5".format(datetime.datetime.now(), val_acc[-1]))

        self.keras_model.save(model_path)

        # save model class to json (for inference)
        self.config.save_model_class(train_classes)

        print("\nTraining completed. Model saved at {}".format(model_path))

        return history


    def decode_predictions(self, preds, top=1):
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            for i in top_indices:
                each_result = []

                # each_result.append(self.CLASS_INDEX[str(i)])
                each_result.append(self.config.decode_classid(str(i)))

                each_result.append(pred[i])
                results.append(each_result)

        return results

    #=========================================
    # Evaluate
    #=========================================
    def evaluate(self, test_dir):
        image_gen_test = ImageDataGenerator(rescale=1./255)
        test_data_gen = image_gen_test.flow_from_directory(batch_size=self.config.BATCH_SIZE,
                                                           directory=test_dir,
                                                           target_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                                                           class_mode='sparse')

        test_labels = (test_data_gen.class_indices)
        test_classes = dict((v,k) for k,v in test_labels.items())

        print("test_classes:  {}".format(test_classes))

        cols = 10
        total = 0
        missed = 0
        missed_info = ""

        for n in range(3) :
            image_batch, label_batch = next(test_data_gen)
            
            total += len(image_batch)
            
            label_batch = label_batch.astype(int)

            predicted_batch = self.keras_model.predict(image_batch)
            predicted_batch = tf.squeeze(predicted_batch).numpy()
            predicted_labels = np.argmax(predicted_batch, axis=-1)

            rows = len(image_batch) // cols + 1
            plt.figure(figsize=(20, 20 * rows // cols))
            plt.subplots_adjust(hspace = 0.4)

            for i in range(len(image_batch)):
                plt.subplot(rows, cols, i+1)
                plt.imshow(image_batch[i])
                plt.axis('off')

                label = label_batch[i]
                label_name = test_classes[label]
                
                predicted_label = predicted_labels[i]
                predicted_name = self.config.decode_classid(str(predicted_label))

                if predicted_name == label_name:
                    color = "blue"
                    plt.title(predicted_name.title(), color=color)
                else:
                    missed += 1
                    color = "red"
                    plt.title("{}\n({})".format(predicted_name.title(), label_name.title()), color=color)

                    missed_info += "[MISSED] {} -> {} \n".format(label_name, predicted_name)

        print("\n=========================================\nEvaluation completed. ")
        print("======{:.2%} correct. \n\nMissed {} out of total {} images:".format((total-missed)/total, missed, total))
        print(missed_info)
        print("=========================================")


    #=========================================
    # classification
    #=========================================
    def mold_input(self, image_size, input_image):
        image_to_predict = KerasImage.load_img(input_image, target_size=(image_size, image_size))
        image_to_predict = KerasImage.img_to_array(image_to_predict, data_format="channels_last")
        image_to_predict = np.expand_dims(image_to_predict, axis=0)
        image_to_predict *= (1. / 255)
        return image_to_predict


    def classify(self, image_path): 
        total = 0    
        
        for root, dirs, files in os.walk(image_path, topdown=False):

            # for artstyle in dirs:
            #     stylepath = os.path.join(root, artstyle)

                # paths = [os.path.join(stylepath, f) for f in os.listdir(stylepath) if os.path.isfile(os.path.join(stylepath, f))]

            paths = [os.path.join(root, f) for f in files]

            # # remove non-image files from the list
            # paths = [p for p in paths if imghdr.what(p)]

            total += len(paths)

            for i, path in enumerate(paths):
                #sourcefile.write("{}\n".format(path))
                # txtfile.write("\n-------Predicting: {} from category '{}'\n".format(os.path.basename(path), artstyle))
                
                #shutil.copy(path, result_dir)

                image_to_predict = self.mold_input(self.config.IMAGE_SIZE, path)

                r = self.keras_model.predict(x=image_to_predict, steps=1)

                # predictions = self.decode_predictions(r, top=int(3))
                predictions = self.decode_predictions(r)

                if i==0:
                    print("\n========== classification result ==========")

                print("=> {:13s} (confidence:{:6.1%}) --- {} ".format(predictions[0][0], predictions[0][1], path))


                # for result in predictions:
                #     print(str(result[0]), " : ", str(result[1] * 100))
                        #txtfile.write("{}: {}\n".format(result[0], result[1]*100))

                    # predictions, probabilities = prediction.predictImage(path, result_count=3) 
                    
                    # for eachPrediction, eachProbability in zip(predictions, probabilities): 
                    #     #print("{}: {}".format(eachPrediction, eachProbability))
                    #     txtfile.write("{}: {}\n".format(eachPrediction, eachProbability))

                    # if self.class_name(predictions[0][0]) != self.class_name(artstyle): 
                    #     missed += 1

                        # missed_dir = os.path.join(missed_base, artstyle)
                        # if not os.path.exists(missed_dir):
                        #     os.makedirs(missed_dir)

                        # shutil.move(path, missed_dir)
                        # #shutil.copy(path, missed_dir)
                        
                        # missed_info += "{}  from {} -> [MISSED] as {} ({})    || {} ({})\n".format(
                        #      os.path.basename(path), artstyle, predictions[0][0], predictions[0][1], predictions[1][0], predictions[1][1])
                        
        # outstr = """\n(end time: {:%Y_%m_%d %H:%M:%S})
        #             \n======= Detection END: {:.2%} correct. Missed {} out of total {} =======
        #             \n\nSummary of the missing:\n{}""".format(
        #             datetime.datetime.now(), (total-missed)/total, missed, total, missed_info)

        # print(outstr)
        # txtfile.write(outstr)

        # # close files
        # txtfile.close()
        # #sourcefile.close()



#=========================================
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and predict art style.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'dataprep', 'train', or 'predict'")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/artstyle_model.h5",
                        help="Path of the model .h5 file")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/artstyle/dataset/",
                        help='Path of the art style dataset')
    parser.add_argument('--epochs', required=False,
                        metavar="epochs for training",
                        help='Number of epochs for training')
    parser.add_argument('--testdir', required=False,
                        metavar="/path/to/artstyle/dataset/",
                        help='Path of the test dataset')
    parser.add_argument('--image', required=False,
                        metavar="path of image(s) to be predicted",
                        help='Path of image(s) for art style prediction')
    args = parser.parse_args()

    # -----------------
    # configuration
    # -----------------
    config = ArtstyleConfig()

    # -----------------
    # executions
    # -----------------
    if args.command == "dataprep":
        assert args.dataset and os.path.isdir(args.dataset), "--dataset path should be a directory"
        
        dataset = ArtstyleDataset(config)
        dataset.prepare()

    elif args.command == "train":
        assert args.dataset and os.path.isdir(args.dataset), "--dataset is needed for training data, and should be a directory"

        model = ArtstyleModel("training", config)

        if args.epochs:
            model.train(args.dataset, epochs=int(args.epochs))
        else:
            model.train(args.dataset)   # use default

    elif args.command == "evaluate":
        assert args.model, "--model (.h5) is needed for evaluation"
        assert args.testdir and os.path.isdir(args.testdir), "--testdir is needed for evaluation, , and should be a directory"

        model = ArtstyleModel("inference", config, model_path=args.model)
        model.evaluate(args.testdir)

    elif args.command == "classify":
        assert args.model, "--model (.h5) is needed for classification"
        assert args.image, "--image directory is needed for classification"
        
        model = ArtstyleModel("inference", config, model_path=args.model)
        model.classify(args.image)

    else:
        print("'{}' is not recognized. " "Use 'dataprep', 'train', 'evaluate' or 'classify'".format(args.command))

