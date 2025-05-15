# DLNLP_assignment_25
## Project Description
This project aims to explore and study NLP tasks. Specifically, we focused on the problem of sentiment analysis. We downloaded a movie review dataset provided by Kaggle and then constructed a balanced dataset containing 30,000 samples for training and testing. This project applyed pre-trained BERT model to extract word embeddings, and built a sentiment classification model that integrates CNN, LSTM and attention mechanism. Ablation studies were also conducted.

## File Overview
The project is organized into the following folders and files:
- **A/** Contains the functions for preprocessing the movie review dataset, tokenization, and embedding.
  - `data_acquisition.py`: Contains function for acquiring the original dataset and saving the preprocessed data.
  - `data_preprocessing.py`: Contains functions for preprocessing the original dataset, including: mapping labels,
checking for empty rows, removing garbled text, removing different languages, tokenizing and embedding the dataset using BERT, and splitting the dataset into training, validation, and test sets.

- **B/** Contains the functions for constructing the models' architectures, model training, testing and experimental results visualising.
  - `modeling.py`: Contains function for creating the dataloader, building baseline and CNN + LSTM + attention mechanism models, training and testing the models, as well as conducting ablation studies.
  - `visualising.py`: Contains functions for visualising the model training process and the experimental results.

- **Datasets**: Stores the datasets that used in this project.
  - `dataset_select.csv`: Stores a reduced version of the original dataset
  - `original_testset.csv`: Stores the testing dataset, with no tokenization or embedding applied.
  - `original_trainset.csv`: Stores the training dataset, with no tokenization or embedding applied.
  - `original_valset.csv`: Stores the validation dataset, with no tokenization or embedding applied.

- **env/** Includes the descriptions of the environment required to run the project
  - `environment.yml`: Defines the environment and its version
  - `requirements.txt`: Lists python packages that required to run the code 

- **figures**: Stores the plots generated during the project, including images from EDA, model training and hyperparameters tuning process as well as the final results.

- **main.py**: The main script that contains the complete workflow code for the sentiment analysis task.

## Required Packages
- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `transformers`
- `langdetect`


## How to Run the Code
1. **Open the terminal and use cd to navigate to the root directory**
2. **Create the Conda Environment:**
   ```bash
   sudo conda env create -f env/environment.yml
3. **Check the Environment:**
   ```bash
   conda info --envs
4. **Activate the Environment:**
   ```bash
   conda activate nlp-final-project-env
5. **Install the required packages:**
   ```bash
   pip install -r env/requirements.txt
6. **Run the main script:**
   ```bash
   python main.py
## Note
- **1.** The project code only includes the results corresponding to the final selected hyperparameters and does not include the run results and visualisation plots produced by the hyperparameters tuning process.
- **2.** Task A presents the training and validation accuracy of the baseline model, while Task B provides the training and validation accuracy of the integrated model.
- **3.** Due to GitHub repository storage limitations, the uploaded dataset has not been tokenized or embedded. However, using BERT for embedding always requires a great amount of time. In actual testing, a full run of the program takes about one hour. Once the embedding and related data have been obtained, the complete execution of the project takes approximately 15 minutes, including training the baseline model, training the integrated model, and conducting ablation studies, etc.