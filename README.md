# High Quality Deep Face
To solve the problem for face recognition can't identify the human real faces and photos of humans. The proposer is proposing a new deep learning architecture model combination that requires to optimize the face detection process and can recognize those valuable images.

## Convolutional Neural Network 
The traditional neural network (CNN) is very popular with many topics such as image processing, image classification,... Because of that power, the proposer wants to apply the technique of CNN for cutting image part and also using [Caffe-DL](https://caffe.berkeleyvision.org/) for reconstructing the platform prediction

<p align="center">
  <img src="read_pic/cnn.png" alt="cnn image"/>
</p>


## Technique Details
* The inspired idea comes from Andrew Ng Baidu Face-recognition enabled entrance [demo](https://youtu.be/wr4rx0Spihs)
* Problem can be solved by training a new CNN model for the only detection but recognizing technique between 2 classes (fake-real). 
* The difference of fake and real features are belong to the image data. Those fakes images can't compare (the size, space, landscape,...) with the real image.

<p align="center">
  <img src="read_pic/process.png" alt="processing image"/>
</p>


## Data-model

* Data has been collected from the internet by Adrian at PyImageSearch
* Model has been trained using the power of Google Colab Server
* The result has been stored into JSON and HDF5 format on Google Drive and can be accessed in [here](https://drive.google.com/drive/u/0/folders/1uvDduzWXzt87VxBjPWFI3fDKakb_ZsIX)











