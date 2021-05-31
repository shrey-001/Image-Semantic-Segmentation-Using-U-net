# Image-Semantic-Segmentation-Using-U-net
This Project uses U-net architecture to build a model to output the segmentation map of the face given the input face image.
### Getting Started
####  System Requirements 
* Numpy
* tensorflow
* keras
* cv2
* os
* time
* Datetime
* tqdm
####  How to Replicate
Upload the Unet.zip file to colab session.</br>
Unet.zip contains(train images,test images, model, logs)
```
!unzip "/content/drive/MyDrive/U-net/u-net.zip" -d "/content/Data"
```
If you want to train the model set, PRETRAINED=False in Colab Notebook else set,
```
PRETRAINED=True
```
Save the predicted result(in .png) in predict_image folder.
Run the Python script F1_score.py to get the F1_score.
```
python f1_score.py
```
### About the data
The Data contains 1999 train images and 100 test images. The labels of image contains different face components like eyebrows, lips, nose,left eye, right eye,etc.
The face components are listed in label_names.txt.</br>
Original Link to the dataset: https://drive.google.com/file/d/1jweX1u0vltv-tYZhYp6mlyDZDy0aDyrw/view


### Preprocessing Data
For every Image we need to do the following:</br>
* Resize all the image to same dimension (256,256,3) using cv2.resize()
* Normalize all the image.
* Convert image to np.float32 numoy array

For every Mask/labels we need to do:</br>
* Read Mask using opencv as a grayscale image (by using cv2.IMREAD_GRAYSCALE)
* Convert every label to binary mask (by using tf.keras.utils.to_categorical).   
* Resize the image to dimension(256,256,1) and use INTER_NEAREST interpolation(to preserve the label during resizing)
### Model

### Result/Analysis
Train Set Accuracy: 97.16%</br>
Validation Set Accuracy: 92.08%</br>
Test Set Accuracy: 90.34%</br>
Train Set loss: 0.0127</br>
Validation Set loss:0.0523</br>
Test Set loss: 0.071</br>
### References
* U-Net: Convolutional Networks for Biomedical Image SegmentationOlaf Ronneberger, Philipp Fischer, and Thomas Brox
* https://cs230.stanford.edu/blog/tensorflow/
