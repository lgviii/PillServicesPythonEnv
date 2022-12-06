-------

# Live Video Pill Inference Testing

The file FULL_PROCESS_VIDEO_prof_color_idea.py contains code that takes a video apart frame by frame. The code then 
attempts to identify objects in the pills utilizing a an SSD MovbileNet V3 model pre-trained on the coco dataset. This
scipt was utilized to find "pills" within live video feeds and draw bounding boxes around them. The pills within the 
bounding boxes are then carved out resulting in a much smaller image with just a small portion of the background in it. 
This smaller image is then passed through both our colors and shapes DNN models and an inference is made and printed to
the screen letting the user know what shape and color the app things the pill is. The video is then pieced back together
with this code resulting in a full length video containing frame by frame pill inferences. 

# Resizing of Pill Images

The file RESIZE_IMAGE_ONE_TIME.py is designed to resize a large batch of pill images into a smaller size. In this case, 
they are resized to size 450x600. This drastically decreased training time of our DNN models. 

# Training code

The file cleaned_up_over_90_percent_random_split_distribution_plot.py contains code utilized to train our final colors DNN model.
This file also shows an experiment we did with the random weighted sampler feature within Pytorch. We essentially plotted out the
distribution of classes created within batches produced by a Pytorch data loader with and without the random weighted sampler. 

# Model Experiment Code

ALL this training was done on a subset of 17827 of the solid color pills from the dataset.

THE CODE in this notebook (FINAL_CLEAN_INFERENCE_DIFFERENT_MODELS.ipynb) illustrats why we chose the pre trained densenet model. In our initial
proposal we had suggested using either vgg16, inceptionv3, or densenet. I was hoping we could
use vgg16 because I had used it in the past and therefore I know how powerful it is. But we were
not able to use it because we didnt have the proper hardware. As the notebook illustrates, 
the code crashed when we tried to train with vgg16, even when freezing the parameters of every 
layer of the model except the last one, which we added. This is becuase our GPU just ran out
of memory. Even when using a 12 gig gpu we could not train on this model. 

The inceptionv3 was good, we were able to load on it and train it. But it just had an accuracy
that was too low. The inception, with all the layers frozen except the last layer, which we 
added, gave us an accuracy of 64.25% after 7 epochs. 

Then we went to the densenet model and froze all of its layers except for 2 linear layers, which
we added. Immediately we noticed validation accuracy was much better. On the validation images
it got up to 82.19% after just 7 epochs. We then took the exact same model, and froze all of the
layers up to layer 300. Training on this model gave us an accuracy of 86.71% after just
7 epochs. 
