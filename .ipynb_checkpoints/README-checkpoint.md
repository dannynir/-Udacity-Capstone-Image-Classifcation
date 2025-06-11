# Image Classification using AWS SageMaker

This project demonstrates how to use AWS SageMaker to train a pretrained model for image classification using profiling, debugging, hyperparameter tuning, and other ML engineering best practices. While the provided dataset is a dog breed classification dataset, this project was implemented using a snake classification dataset with 5 snake classes.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
I used a snake classification dataset with 5 classes instead of the dogbreed classification dataset
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
The dataset was uploaded to an S3 bucket to allow SageMaker access during training and inference.

## Hyperparameter Tuning
I chose to finetune the resnet50 model due to its relevance to image classification. The data had 5 classes, so I attached a linear layer with outcome size of 5. I chose to tune hyper parameters related to learning rate in the range from 0.0001 to 0.01 and batch size in the range of [32, 64, 128]

Screenshot of completed training jobs:

![Training Jobs](./Output%20Images/Training%20Jobs.png)  

The one named ‘snake-classifier-‘ is the training job with best parameters. The others are hyperparameter training jobs

- Logs metrics during the training process
![Training Logs](./Output%20Images/Training%20Logs.png)  


![Hyper Parameter Tuning Logs](./Output%20Images/Logs%20Hyperparameter%20tuning.png) 
 
- Tune at least two hyperparameters
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

Profiler html/pdf file can be found at "CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/ProfilerReport/profiler-output/profiler-report.html"

### Results
What are the results/insights did you get by profiling/debugging your model?
As seen above the training loss decreased with each step which is a good 
Other rules like vanishing gradients or overfitting did not trigger alerts.



## Model Deployment
Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Since sagemakers default handlers did not work for image input I had to define my own inference.py to tweak the model_fn(),input_fn(), predict_fn() and output_fn() with my own custom functions to transform the input image and also the serve the outputs in an understandable way. I also had to define the serializer for reading image input and a reserialize to output a string since the defaults threw an error. This enabled the endpoint to receive .jpg images and return class label predictions. With all this the model was deployed to the following endpoint in S3:

![Deploy Code](./Output%20Images/Deploy.png)

Endpoint:
![Endpoint Image](./Output%20Images/Deployed%20Endpoint.png)

Querying from Endpoint:
![Query Endpoint](./Output%20Images/Deploy%20Query.png)

![Final Output](./Output%20Images/Deploy%20query%20op.png)

