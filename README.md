# Machine Learning for Genomics: Project 1
## Table of contents
1. [How to setup this repo](#setup-and-installation)
2. [Reproduce experiments](#reproduce-experiments)
3. [File structure](#file-structure)
3. [Contributors](#contributors)
## Setup and Installation
### Data setup
After you have downloaded the code you need to download the neccesary data for the experiments and place it in the proper folder:

1. Create the folder data in the root directory of this repo. You can do this using the command `mkdir data`.
2. Download all the files from the [project polybox]( https://polybox.ethz.ch/index.php/s/7ooTHUEd888N4FL) at once. This should result in the file `ML4G_Project_1_Data.tar` being placed in your `~/Downloads` folder.
3. Move the file `ML4G_Project_1_Data.tar` from your `~/Downloads` folder into the `data` folder created in the step 1. 
3. Unzip the file `ML4G_Project_1_Data.tar`. This should result in your repo having the following structure: `your_repo_root/data/ML4G_Project_1_Data`.
4. Now go into the recently created folder `your_repo_root/data/ML4G_Project_1_Data` and unzip all the `.zip` files that you may find there.
5. Inside the folder `data` create an empty folder named `numpy`. This can be done by running the command `mkdir data/numpy` from the root of the repo. The existence of this folder is needed to run some of the code of the repo.

### Environment setup
To properly run the code of this project you need Python 3.12 installed in your machine. We recommend using an environment for the project, for example using [CONDA](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html). You can then install the required packages running the following command from the root folder of the repository:

```bash
pip3 install -r requirements.txt
```

[*Troubleshooting Note*] If you are facing issues with the installation you may want to remove the lines related to the installatio of PyTorch (torch, torchaudio and torchvision) from the `requirements.txt` file. Then follow the instructions from the [PyTorch documentation](https://pytorch.org/get-started/locally/) to properly install these packages for your device.

## Reproduce Experiments
The first thing you will need to do is to generate files with the needed features and target values to train the predicor model. To do this run the following commands from the root of the repo:

```bash
python3 src/generate_numpy.py X1_train
python3 src/generate_numpy.py X1_val
python3 src/generate_numpy.py X2_train
python3 src/generate_numpy.py X2_val
python3 src/generate_numpy.py X3_test
```

After completing this step you should find that the following files had been created inside the `your_repo_root/data/numpy` folder: `X1_train_X.npy`, `X1_train_y.npy`, `X1_val_X.npy`, `X1_val_y.npy`, `X2_train_X.npy`, `X2_train_y.npy`, `X2_val_X.npy`,   `X2_val_y.npy`, `X3_test_X.npy`, `X3_test_y.npy`.

Now you are ready to train the predictor model and to make predictions. This is done by running the following command from the root of the repo:

```bash
python3 src/train.py
```

When developing this project the training was performed on an Apple device using metal acceleration for Apple Silicon. However, the training script will first try to use CUDA acceleration if it is available for your device. If CUDA is not available, the script tries to use acceleration for Apple silicion. Lastly, if none of the two hardware acceleration is available the script will just run using the CPU.

After completing this step you should find that the following file has been created in the root of the repo: `submission.csv`. This file contains the predictions over the test set in the appropiate format to be submitted for evaluation on the online grader. 

## File Structure
The relevant file strucutre of the project along which what can be found in each file is listed here.
* `./configuration`: folder storing configuration files.
    * `config.json`: configuration file with hyperparameters for the model and training instrucitons.
* `./data`: folder for data storage.
    * `ML4G_Project_1_Data`: data originally provided for the project.
    * `numpy`: features and target values for training/evaluating the model stored in `.npy` format.
* `./notebooks`: folder for jupyter notebooks.
    * `analysis.ipynb`: jupyter notebook to get data insights.
* `./src`: source code for the project.
    * `./base_model.py`: parent class common for all prediction models in case multiple strategies are implemented.
    * `./cnn_model.py`: CNN model used for the project.
    * `./generate_numpy.py`: generate features and targets in an appropiate format to train the models.
    * `./train.py`: train a model and make predictions on the test set (generates the submission file).
    * `./utils.py`: miscelaneous functions that may be shared among scripts.
* `./requirements.txt`: python package requirements to run the code.



## Contributors
- Julen Costa Watanabe
- Juan Garcia Amboage
