import segmentation_models as sm
from PIL import Image
import glob
import numpy as np
import cv2

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# load your data
raw_image_list = []
truth_image_list = []
for filename in glob.glob('Human-Segmentation-Dataset/Training_Images/*.jpg'): #assuming gif
    #im=Image.open(filename).convert('RGB')
    raw_image_list.append(cv2.imread(filename))

for filename in glob.glob('Human-Segmentation-Dataset/Ground_Truth/*.png'): #assuming gif
    #im=Image.open(filename).convert('RGB')
    truth_image_list.append(cv2.imread(filename))

print('RAW IMAGES')
print(len(raw_image_list))
print('Thruth IMAGES')
print(len(truth_image_list))

x_train = raw_image_list[:-round(len(raw_image_list)*0.2):]
print('XTrain:' + str(len(x_train)))
y_train = truth_image_list[-round(len(raw_image_list)*0.2):]
print('YTrain:' + str(len(y_train)))
x_val = raw_image_list[:-round(len(truth_image_list)*0.2):]
print('Xval:' + str(len(x_val)))
y_val = truth_image_list[-round(len(truth_image_list)*0.2):]
print('Yval:' + str(len(y_val)))

#1). X_train - This includes your all independent variables,these will be used to train the model, also as we have specified the test_size = 0.4, this means 60% of observations from your complete data will be used to train/fit the model and rest 40% will be used to test the model.
#2). X_test - This is remaining 40% portion of the independent variables from the data which will not be used in the training phase and will be used to make predictions to test the accuracy of the model.
#3). y_train - This is your dependent variable which needs to be predicted by this model, this includes category labels against your independent variables, we need to specify our dependent variable while training/fitting the model.
#4). y_test - This data has category labels for your test data, these labels will be used to test the accuracy between actual and predicted categories.

# x_train, y_train, x_val, y_val = load_data(...)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

n = np.stack(x_train, axis=0)
print(n)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(
   x=np.array(x_train),
   y=np.array(y_train),
   batch_size=16,
   epochs=100,
   validation_data=(np.array(x_val), np.array(y_val)),
)