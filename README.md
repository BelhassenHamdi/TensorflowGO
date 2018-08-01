# TensorflowGO
WIP : This repository contains a mini project written in Golang. It tackles the problem of deploying a tensorflow model in production. This work is still in progressm too many other modules will be added to the project to be more user friendly and easily deployed by novice programmers. I tried to make all the development with Go as much as possible in the aim to make it more comprehensible fo go developpers. I made all the image processing and normalisation using golang and not Tensorflow OP library. Feel free to make suggestions and comments.
Work to be done:
  - Add the work to docker container and make it as an AI server waiting for image requests
  - Make a dockerfile for easy deployment
  - Add visualization
  - Print the best 3 labels
  - Make the model gets a Tensorflow model directory path (not freezed yet eg *.meta and *.data files) and generates the .pb file.
    it will also retrieve all the valuable informations (the input and output for the model and the sizes) and make all the rest of the work automatically.
This will certainly help programmers not used with golang or Tensorflow to be more at aise when dealing with deploying models in production.
  
  
## To test this project:
  I made the project download the model itself, but if you already have the model in your machine all you have to do is to set the path for the model as a first argument.
  The second argument for running the inference is the images dataset path. I made the model gets as input a batch of five images
  I will change that asap and make it as a manual setting as an argument for the model. For now, if you do not set the images path 
  the model will go and look for /data/images directory.
  I will make th
