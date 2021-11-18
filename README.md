Name: Catherine Gai

SID: 3034712396

Email: catherine_gai@berkeley.edu

Link to project report website: [Here](https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj5/cs194-26-aay/catherine_gai_proj5/Project%2005.html). 

This folder contains four functional python (and .ipynb) files: "load_nosetip_data.py", "nosetip_detect.py", "proj5_part2.ipynb", "proj5_part3.ipynb". 

In addition, the folder contains "catherine_gai_proj5.zip" file, which contains a project report and all generated image predictions, in case the upload utility does not work. 



**load_nosetip_data.py:**

This python file contains classes and functions that generates a training dataset and validation dataset for nosetip detection. Both training and validation dataset inherits the `torch.Dataset` class. 240 images are pre-loaded, and splitted into 192 training data and 48 validation data. 

* training_dataloader(*params*): Given batch size and rescale factor, the function returns a `torch.Dataset` object and a `torch.DataLoader` object, containing images rescaled to $(480 \times \text{rescale factor}, 640 \times \text{rescale factor})$, divided into batches for future training. 
* validation_dataloader(*params*): Given rescale factor, the function returns a `torch.Dataset` object and a `torch.DataLoader` object, containing images rescaled to $(480 \times \text{rescale factor}, 640 \times \text{rescale factor})$. The function will load all validation data in one go. 



**nosetip_detect.py:**

This python file contains a NosetipNet neural network, and commands that generate required images. 

Detailed architecture of neural network: 

I used 3 convolutional layers, the first with kernel size 3, the second and third with kernel size 5, and 2 fully connected layers. The channels of convolutional layers are 4, 8 and 12 respectively. The first fully connected layer is followed by a batch-normalization. 



**proj5_part2.ipynb:**

This .ipynb file contains all functions and commands that set up a neural network, train the neural network and visualize results, loss curve, and convolutional filters. 

* rotate(*params*): rotate all images in a batch with specified angles, and adjust feature points accordingly. The function will return two lists: x_rotated_lst where each element is a batch rotated at a particular angle, y_rotate_lst where each element is a batch of corresponding rotated features. The feature locations are represented by fractions (the proportion of a pixel location in total image width/ height). 
* shift_vertical(*params*): shift all images in a batch vertically with specified pixels, and adjust feature points accordingly. The function will return two lists: x_shifted_lst where each element is a batch shifted at a particular pixel value, y_shifted_lst where each element is a batch of corresponding shifted features. The feature locations are represented by fractions (the proportion of a pixel location in total image width/ height). 
* shift_horizontal(*params*): shift all images in a batch horizontally with specified pixels, and adjust feature points accordingly. The function will return two lists: x_shifted_lst where each element is a batch shifted at a particular pixel value, y_shifted_lst where each element is a batch of corresponding shifted features. The feature locations are represented by fractions (the proportion of a pixel location in total image width/ height). 



**proj5_part3.ipynb:**

This .ipynb file contains all functions and commands that set up a slightly modified ResNet18 neural network, train the neural network and visualize results, loss curve, and test on additional images. Running the file will also generate a .csv file that contains predicted features on test images. 

* rotate(*params*): rotate all images in a batch with specified angles, and adjust feature points accordingly. The function will return two lists: x_rotated_lst where each element is a batch rotated at a particular angle, y_rotate_lst where each element is a batch of corresponding rotated features. The feature locations are represented by true pixel values (the true coordinate of a point). 
* shift_vertical(*params*): shift all images in a batch vertically with specified pixels, and adjust feature points accordingly. The function will return two lists: x_shifted_lst where each element is a batch shifted at a particular pixel value, y_shifted_lst where each element is a batch of corresponding shifted features. The feature locations are represented by true pixel values (the true coordinate of a point). 
* shift_horizontal(*params*): shift all images in a batch horizontally with specified pixels, and adjust feature points accordingly. The function will return two lists: x_shifted_lst where each element is a batch shifted at a particular pixel value, y_shifted_lst where each element is a batch of corresponding shifted features. The feature locations are represented by true pixel values (the true coordinate of a point). 

