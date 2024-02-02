Stage 2 of Deep Learning Projects

This stage aims to help students get familiar with PyTorch, one of the most popular toolkit used in deep learning, and scikit-learn (sklearn), 
one of the most popular toolkit used in classic machine learning. Please write your first program with PyTorch to implement the MLP model introduced in class.

2-1: Download the dataset provided by the instructor, which covers both training set and testing set. Take a look at the dataset before writing your code.

2-2: Write your own code based on the template for stage 2 (First, copy the code from stage 1 folder into stage 2, and also change the import commands accordingly. 
You can reuse/make changes to the code provided in Stage 1, e.g., write a new Dataset_Loader for your new dataset, change the training/testing set partition code 
in the Setting (no train/test set split or cross validation will be needed for stage 2), change the Method_MLP architecture to get adapted to the new input/output, 
and include more evaluation metrics … , define more evaluation metrics, e.g., F1, Recall, Precision, we are doing multiclass classification here, so the binary F1/Recall/Prec cannot be used, 
you can use the “weighted”/”macro”/”micro” version of the metrics instead).

2-3: Train the MLP model with the training data, and apply the learned model to the testing set. Generate the learning curve plots, and report the final evaluation result.

2-4: Change the model architecture with more layers, with different loss functions, different optimizers and other settings to increase the performance score.

2-5: Write a report about your experimental process and results based on the provided report template (5 pages at the maximum, longer reports will be “penalized” :( ).

2-6: (Optional): If your computer GPU supports CUDA programming and you can also try to run your model with your GPU with examples like https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
