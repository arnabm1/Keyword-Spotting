This reporsitory trains quantized-scaled models using Qkeras for Keyword Spotting using the Google Speech Commands dataset.
Implementation of the model training for the **tinyML 2022 paper "[A Fast Network Exploration Strategy to Profile Low Energy Consumption for Keyword Spotting](https://arxiv.org/abs/2202.02361)"**

**Description of files:**

**prepare_dataset:** Generates Json compatible file to be fed to the model from the raw audios

**utils:** Contains helper functions for model training

**q_train:** training code to generate quantized-scaled models

**Steps to run the code**

1. Generate a conda virtual environment with "**conda create -n envname python=3.9**".
2. Run the 'requirements.txt' file with "**pip install -r requirements.txt**".
3. Download the dataset with "wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz".
4. Save the dataset on the drive and specify the dataset location on the 'prepare_dataset.py" file.
5. Run the 'q_train.py' file with "**python q_train.py**" to create and save trained models on the "Trained_models" directory.  
