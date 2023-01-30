# $BP(\lambda)$
An online, BPTT-free method for learning and using synthetic gradients.

## Dependencies
Beyond the standard python libraries, you will need [Pytorch](https://pytorch.org/) (we use version 1.7.0, but later should work) to define the neural network models and as well as [ignite](https://github.com/pytorch/ignite) (version 0.4.2) which wraps the training regime. [Sacred](https://github.com/IDSIA/sacred) (version 0.7.4) is used to record experiment details.

## Steps to run 
Scripts to run the tasks presented in the paper can be found in the /scripts folder. For a given script:
1. Define model/training/dataset hyperparameters*. These can be changed in the respective file in the /configs folder.
2. Define the OBSERVE variable (preset to False). If true, then experiment results will be saved in a generated /sims folder.
3. Run the script.
