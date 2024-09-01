# Applying NLP to Twitter Airline Sentiment to Reveal Significant Text Features

## Dataset description
The Twitter US Airline Sentiment dataset on Kaggle contains 14640 tweets directed at major U.S. airlines. This dataset is widely used for training and evaluating models in sentiment analysis, offering insights into public perception of airline services. Each tweet is labeled with sentiment (positive, neutral, or negative) and provides additional metadata such as the airline mentioned, text of tweet, reason for negative tweet, sentiment of that tweet... In our work, we are using  only the literal text and the label.

## Project description
In recent years, flying across the world has become increasingly accessible, making it essential for travelers to understand the benefits and drawbacks of various major airlines. With Twitter emerging as a prime platform for sharing airline reviews, analyzing a vast collection of tweets offers valuable insights into public sentiment towards airlines like Southwest, Frontier, and United. This project aims to predict the sentiment of tweets—whether positive, negative, or neutral—using the "Twitter US Airline Sentiment" dataset from Kaggle.

### Challenges and Approach:

When applying NLP techniques to airline sentiment analysis, two primary challenges arise:

Data Preprocessing: Extracting meaningful features from text is crucial for accurately capturing the sentiment expressed in tweets. This project tackles the challenge by experimenting with different preprocessing techniques (for an example removing mentions, tags, stop words, converting emojis and emoticons to text...) , each emphasizing various aspects of the text to optimize the sentiment analysis process.

### Models:

**BERT with a Fully Connected Layer:** The first model utilizes BERT's tokenizer and base model, with an additional fully connected layer for sentiment classification.

**BERT with Multiple Fully Connected Layers:** The second model builds upon BERT by adding several fully connected layers to enhance the model’s ability to capture sentiment nuances.

**CNN Model:** The third model creates a custom tokenizer using the NLTK library and implements a three-way Convolutional Neural Network (CNN). This model processes text using multiple kernels, combining their outputs into a fully connected layer for sentiment prediction.

### Training:
The training process will utilize a 90-5-5 split for training, validation, and test sets. First two
models will be trained using Adam optimization, and for third we are using Adadelta and all three models are using cross entropy loss as we are working with multi
classification. While the pretrained model was trained for 10 epochs, each of the other models were
trained for 66 epochs with a learning rate of 1e-6. Furthermore, the pretrained model was trained
with a mini-batch size of 5 while the other two models were trained with a batch size of 64.


## Instructions for Running the Project
### Required Packages:
nltk emoji transformers pyspellchecker tqdm

### Environment Setup
1. Python Version: Ensure you have Python 3.8 or higher installed.
2. Install Packages: `!pip install nltk emoji transformers pyspellchecker tqdm`
3. Data Download: Download the dataset from Kaggle using this [link](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
4. Launch Jupyter Notebook: Start the Jupyter Notebook server by running: `jupyter notebook`
