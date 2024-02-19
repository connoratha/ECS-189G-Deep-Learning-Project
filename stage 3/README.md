This stage aims to help students get familiar with the convolutional neural network (CNN) model, and try to use the CNN model to classify image data for object recognition.

3-1: Download the three image datasets provided by the instructor, one about hand-written digit numbers, one about human faces and one about colored objects. 

(For the human face dataset, it is converted to gray-scale image from colored images by assigning R/G/B with equal numbers. So its data also has three channels (R, G, B) but with equal pixel values for the R/G/B in these three channel matrices. You don’t need to use three channel matrices, use one channel should be sufficient.)

3-2: Write your own CNN model in the provided code template.

3-3: Train a CNN model with the hand-written digit images, and apply it to recognize the digit images in the testing set. Generate the learning curves and report the evaluation results.

3-4: Train two other CNN models with the face images and colored objects, and apply them to recognize the faces and color objects in the testing set, respectively. Generate the learning curves and report the results.

3-5: Try to change the configurations of the provided CNN model (with different model depth, kernel size, padding, stride, pooling layer, hidden layer dimension, different loss function, etc.), and report the results to see the impacts of configuration on the model performance.

3-6: Write a report about your experimental process and results based on the provided report template (5 pages at the maximum).

3-7: (Optional): If you have GPUs supporting CUDA programming, you can also try to run the CNN model with your GPU instead.
