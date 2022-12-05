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