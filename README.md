### A-Probabilistic-Visual-Question-Answering-Model-Based-VQA
Visual Question Answering (VQA) is a system which takes image as an input and a question about the image and generates a natural language answer using complex reasoning. Thus, a VQA needs detailed understanding of the image and complex reason to predict the answer. This project introduces a pretrained model (VGGNet) to extract image features and Word2Vec to embed the words and LSTM to get word features from the question and after combining the results will predict the answer having highest probability.


### Required Packages

spacy<br />                     tensorflow==2.2.0<br />
Keras==2.3.1<br />              random<br />
pickle<br />                    gc<br />
operator<br />                  collections<br />
itertools<br />                 numpy<br />
pandas<br />                    scipy<br />
scikit-learn<br />              matplotlib<br />

### Process Flow Diagram/ Model Prototype:

![image](https://user-images.githubusercontent.com/61022065/114544743-0b2ca700-9c29-11eb-9e7c-6caf6382d05e.png)

The model basically takes as an input and an image form the dataset, then we extract text features and image features and after combining the features predicts the answer with highest probability. For questions, we extract features using Word2Vec. For the images, we input the feature vectors trained from pretrained VGG(net) to the model and extract image features which we use furthur for training. During the training, we keep saving weights after five epochs to a '.hdf5' file which can be used for furthur training the model by using load_weights method. For answers, we use 'Label Encoder' to get the encoded labels which will be used while testing. For testing the model, we compare the predicted class to a list of answers returned by encoding labels and then predict the answer with highest probability. To predict the answer we use a function which on the basis of treshold limit, returns tuple of lists of filtered questions, answers and imageIDs based on the frequency of occurance of the answer.

(Word2Vec +LSTM) Model --> Image model: input image vectors --> hidden layers --> activation layer --> dropout layer
                       --> Question model: input question vectors --> LSTM layers --> hidden layers --> activation layer --> dropout layer
                       
### To train and test the (Word2Vec + LSTM) VQA model do the following steps
1. Move into A-Probabilistic-Visual-Question-Answering-Model-Based-VQA directory
2. Download the image features pickle file for train images from:
   "https://drive.google.com/drive/u/0/folders/1IL7y2FRBEkyMIJaCg6JXPE1wmRo2bthT"
3. Download the image features pickle file for validation from:
   "https://drive.google.com/drive/u/0/folders/1Y68VB3hI5Jq8lmSXyGTXtlYtVilIBOMG"
4. Download the preprocessed data from the directory "Pre-processed files" and save them in the same folder.
5. Install the requirements using the command "pip install -r requirements.txt"
6. Run the VQA_Word2Vec_LSTM.py to train and test
                       
### You can download the pickle files used for training and testing here:

"https://drive.google.com/drive/u/0/folders/1Y68VB3hI5Jq8lmSXyGTXtlYtVilIBOMG"






