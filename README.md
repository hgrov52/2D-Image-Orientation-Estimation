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

The cascade of a classification network with 4 outputs feeding into a regression network as unstable results at this point, most likely due to loss of target angle through two separate neural networks at once. This is a good outlet for future work, because as standalone networks both show promising results. 
