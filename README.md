# 2D-Image-Orientation-Estimation

### Defaults
--batchSz 60 | (Any int)

--nEpochs 40 | (Any int)

--type regression

--nClasses None (initialized based on --type)

--data-type turtles

### To Train
python3 train.py [options]

--type can be "classification", "classification4", "classification8", "regression", "regression45"

--data-type can be "turtles" or "mnist"

For example, training on the mnist dataset with a 4 output classification network 

python3 train.py --data-type mnist --type classification4

To train a network for the first time, or to override current network weights, use --no-resume

To see example outputs of a network, use --example

To see error statistics and distribution or a network, use --test


The cascade of a classification network with 4 outputs feeding into a regression network as unstable results at this point, most likely due to loss of target angle through two separate neural networks at once. It can be run, cascade.py, with no options, however it is not ready for testing. This is a good outlet for future work, because as standalone networks both show promising results. 

## Code 
Most of the action is happening in train.py. Data is loaded in the file data.py, and the testing results are plotted from utils/test.py. All image augmentation happens in data.py, with some math coming from utils/bbox_utils.py as well as util_functions.py. 

The DenseNet implementation is located at the bottom of train.py.

On first run the data.py file will pre-process the images and save the to a pickle file for next run. 


