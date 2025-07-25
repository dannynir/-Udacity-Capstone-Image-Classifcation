# Udacity Capstone: Snake Image Classification using AWS SageMaker

The core problem tackled in this project is to develop an automated system that can accurately identify a snake's species from a photograph. Formally, this is structured as a multi-class image classification problem in the field of computer vision and machine learning. The retrained model is deployed as a web application on AWS (Amazon Web Services). In the final application, a user can upload a snake photo, and the system will return the predicted species of the snake

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

##Software and libraries
Flask
pytorch
Sagemaker
smdebug
boto3
torch
torchvision
numpy
Pillow
requests
flask
pandas

## Dataset
The dataset used in this project was sourced from the Kaggle “Pre-processed Snake Images” collection, featuring 1,300 labeled images representing five snake species: Northern Water Snake, Common Garter Snake, Dekay’s Brown Snake, Black Rat Snake, and Western Diamondback Rattlesnake. Each image was preprocessed for size and quality and resized to 384x384 pixels. The total dataset size is about 1.1 GB. The data was pulled into jupyter notebook as follows:

import kagglehub

# Download latest version
path = kagglehub.dataset_download("sameeharahman/preprocessed-snake-images")

### Access
The dataset was uploaded to an S3 bucket to allow SageMaker access during training and inference.So running the notebook should automatically pull the data and upload to S3

## Hyperparameter Tuning
I chose to finetune the resnet50 model due to its relevance to image classification. The data had 5 classes, so I attached a linear layer with outcome size of 5. I chose to tune hyper parameters related to learning rate in the range from 0.0001 to 0.01 and batch size in the range of [32, 64, 128]

Screenshot of completed training jobs:

![Training Jobs](./Output%20Images/Training%20Jobs.png)  

The one named ‘snake-classifier-‘ is the training job with best parameters. The others are hyperparameter training jobs

- Logs metrics during the training process
![Training Logs](./Output%20Images/Training%20Logs.png)  


![Hyper Parameter Tuning Logs](./Output%20Images/Logs%20Hyperparameter%20tuning.png) 
 
- Tune the two hyperparameters
![HPO Parameters](./Output%20Images/HPO%20Ranges.png)  

- Retrieve the best best hyperparameters from all your training jobs
![Best HPO](./Output%20Images/Best%20HP.png)  


## Debugging and Profiling
Using smdebug library debugging and profiling was done. The following rules were added
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport())
]
collection_configs = [
    CollectionConfig(
        name="train_loss",  # Custom collection for training loss
        parameters={"include_regex": ".*CrossEntropyLoss_output.*", "save_interval": "10"}
    ),
    CollectionConfig(
        name="eval_losses",  # Custom collection for evaluation loss
        parameters={"include_regex": ".*CrossEntropyLoss_output.*", "save_interval": "10"}
    )]
Hooks were also added to train_model.py

The results were as follows

![Debug Result](./Output%20Images/Debug%20Training%20loss.png)

Profiler html/pdf file can be found at "CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/profiler-report.html"



## Model Deployment
In addition to the SageMaker endpoint, a more customized deployment was done to demonstrate a complete end-to-end application with a user interface using docker and Flask API based web app. EC2 instance was utilizing to run this application

1.The UI was launched with the following steps:
2.Need the .pem file from EC2 in same folder as the terminal
3.Run chmod 400 <my-key>.pem to set correct permissions
4.SSH into EC2 instance using : ssh -i capstone.pem ec2-user@98.81.229.174
5.Once into EC2 terminal run this:
cd ~/web_app
pip3 install -r requirements.txt
sudo python3 application.py

The UI needs to be run within EC2 terminal as "sudo python3 application.py" at which point the UI will launch at http://98.81.229.174/ 

Docker Container pushed to ECR:
![ECR](./Output%20Images/ECRDeployedmodelDocker.png)

Endpoint (from Docker container in ECR):
![Endpoint Image](./Output%20Images/DockerEndpoint.png)

UI (at launch):
![UI Launch](./Output%20Images/WebAppUI.png)

UI (Model Results):
![UI Launch](./Output%20Images/WebAppUIresults.png)


