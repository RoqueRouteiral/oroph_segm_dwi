# The role of Diffusion Weighted Imaging for oropharyngeal primary tumor segmentation with deep learning
The goal of the project is to asses the role of DWI for oropharyngeal primary tumor segmentation.
As a secondary goal, we implemented a two-stage approach that outperforms the implementation of its one_stage counterpart.

# Walkthrough of the code

The configuration file can be found in config.py. It can be used to change the hyperparameters for training or inference.

The main file can be found in train.py. It is used to run the experiments as defined in the configuration file.

Inside the directory "tools" you can find the scripts needed during the training:

Model_factory: Script that loads the models and performs training, prediction and training
loaders: loaders*.py. Each loader is used differently depending on the config file.
- loader_mp.py is to run the full end-to-end approaches (UNet, YNet, AlfaNet).
- loader_mp_boxes.py to run the segmentation stage from the two-stage approach
- loader_mp_boxes_inf.py to run the segmentation stage in inference time from the boxes define in the config file
- loader_double_gt_mp.py for the XNet.

