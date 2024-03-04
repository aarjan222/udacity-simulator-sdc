## Intro
# End to End Learning for Self-Driving Cars
This project is a self driving car made in simulator using convolutional neural networks(CNNs) to directly map raw pixels from a three camera facing infront of the car with different field of view to steering commands.


## Key Features
- Implementation of CNN Architecture for end-to-end learning in self driving cars.
- Augementation of training data with with artificial shifts and various transformations.


## Required
- Download simulator car from udacity github repository named "self-driving-car-sim" according to your OS. Read readme.md file for installation process.  
- make your own virtual environment  
```shell
pip3 install -r requirements.txt
```

## Running
- Open the simulator car
- ```shell
python3 drive.py
```
- Run the car in autonomous mode


## Results of different model
There are different model which I had trained on different ways. Here are some of them and its results are also discussed below:

- model1
    - trained on augmented image, and 4 layers of dropout
    - works good for validation data, almost like overfitting


- model2
    - trained on augmented image, but no dropout layers 
    - works very good for validation data
    - not much good for training data


- model3
    - trained on augmented image, no dropout layers
    - but very low learning rate=0.0001 with Adam