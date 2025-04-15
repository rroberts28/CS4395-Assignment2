# CS4395 Assignment 2
https://github.com/rroberts28/CS4395-Assignment2   
Richard Roberts  
rjr230003

## 1 Introduction and Data

This project explores the usage of two types of neural network for the task of sentiment analysis. We are given training and validation sets that each contain a number of Yelp reviews, each with a corresponding star rating. The two types of neural network we will be comparing against this task are the feed-foward neural network and the recurrent neural network, we would like to determine which model performs better at analyzing the sentiment within these Yelp reviews.

To accomplish this we evaluated the performance of each model after training with different hidden layer dimensions (10, 50, 100, 200) and for different numbers of epochs (5, 10, 20). Interestingly, the feed-forward neural network consistently outperformed the recurrent neural network, showing that for certain tasks with specific features, sometimes the simpler model is better suited to the task.

The data set used for this project was split into training and validation sets, but there also appeared to be variants of these sets in the Data_Embedding.zip file. The training set in the zip file was equally balanced with 3200 reviews of each star rating. Its corresponding validation set only contained 320 1 star reviews, 320 2 star reviews and 160 3 star reviews, with no other reviews of other ratings. Whereas the training and validation set found in the top level directory (and was predominantly used during experimentation) were different. That training set was unbalanced and contained 3200 reviews of both 1 and 2 star reviews and 1600 3 star reviews. Its corresponding validation set was unbalanced as well, in the same way, containing 1/10 the number of reviews found in the training set.

**./{training,validation}.json - Sets used in experimentation**
| Rating        | Training | Validation |
|---------------|----------|------------|
| 1.0 stars     | 3200     | 320        |
| 2.0 stars     | 3200     | 320        |
| 3.0 stars     | 1600     | 160        |
| 4.0 stars     | 0        | 0          |
| 5.0 stars     | 0        | 0          |
| Total reviews | 8000     | 800        |

**./Data_Embedding/{training,validation}.json**
| Rating        | Training | Validation |
|---------------|----------|------------|
| 1.0 stars     | 3200     | 320        |
| 2.0 stars     | 3200     | 320        |
| 3.0 stars     | 3200     | 160        |
| 4.0 stars     | 3200     | 0          |
| 5.0 stars     | 3200     | 0          |
| Total reviews | 16000    | 800        |

After experimenting with the zipped up training/validation.json files, the important finding remains the same, the FFNN outperforms the RNN.

## 2 Implementations

### 2.1 FFNN

For the FFNN implementation, I needed to complete the forward method to be able to process inputs through the neural network.

```python
def forward(self, input_vector):
    # obtain first hidden layer representation
    hidden = self.activation(self.W1(input_vector))
    
    # obtain output layer representation
    output = self.W2(hidden)
    
    # obtain probability dist.
    predicted_vector = self.softmax(output)
    
    return predicted_vector
```

The convert_to_vector_representation creates a bag-of-words representation from the data, which is the input vector into the forward method. To obtain the hidden layer representation, we first apply the linear transformation 'W1'

```W1 = nn.Linear(input_dim, h)```

which takes the vector from dimension 'input_dim' to 'h', we then apply the ReLU non-linear activation

```activation = nn.ReLU()```

to each element of the transformed vector. To obtain the output layer we apply the second linear transformation 'W2' to the hidden layer representation, taking it from dimension 'h' to the 'output_dim'. Finally to obtain the probability distribution we simply apply the LogSoftmax function 'softmax' to the output layer representation.

The initial network weights and biases are set using:

```random.seed(42)```   
```torch.manual_seed(42)```

and the optimizer is using Stochastic Gradient Descent with a momentum of 0.9 and a learning rate of 0.01.

```optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)```

Unlike the RNN implementation there is no early stopping in the FFNN.

### 2.2 RNN

Similarly to the FFNN, four (only three required) sections needed to be filled in order to complete the forward method. Unlike the FFNN which uses the bag-of-words representation, the RNN processes words sequentially and uses the pre-trained word embeddings. So to obtain the hidden layer representations we utilize the PyTorch built-in RNN module and "Apply a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence." Here we are using tanh non-linearity. We then squeeze the hidden layer to remove the first dimension that was representing the number of layers in the RNN. Next we apply a linear transformation to obtain scores for each of the 5 star ratings. Finally we use softmax in a similar way as before, to obtain the probability distribution from these scores.

```python
def forward(self, inputs):
    # obtain hidden layer representation
    _, hidden = self.rnn(inputs)
    
    # obtain output layer representations
    output = self.W(hidden.squeeze(0))
    
    # obtain probability dist.
    predicted_vector = self.softmax(output)
    
    return predicted_vector
```
This RNN implementation is also using the Adam optimizer instead of SGD and stops training early when validation accuracy starts to decrease to prevent overfitting.

https://pytorch.org/docs/stable/generated/torch.nn.Linear.html   
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

## 3 Experiments and Results

To evaluate both the FFNN and RNN I used training and validation accuracy as the primary metric, which are both tracked by the provided implementation of the neural networks. For both models I monitored both the training and validation accuracy while varying the hidden layer dimension (10, 50, 100, 200) and the epochs for the FNN specifically as the RNN stops early (5, 10, 20).

### FFNN Results

| Hidden Dim | Epochs | Training Accuracy | Validation Accuracy |
|------------|--------|-------------------|---------------------|
| 10         | 5      | 64.08%            | 57.00%              |
| 10         | 10     | 75.95%            | 60.25% (epoch 9)    |
| 10         | 20     | 81.99%            | 61.13% (epoch 16)   |
| 50         | 5      | 65.10%            | 59.75%              |
| 50         | 10     | 78.03%            | 61.00% (epoch 6)    |
| 50         | 20     | 93.69%            | 61.75% (epoch 18)   |
| 100        | 5      | 64.55%            | 60.25%              |
| 100        | 10     | 79.25%            | 61.38% (epoch 9)    |
| 100        | 20     | 91.78%            | 62.63% (epoch 12)   |
| 200        | 5      | 65.91%            | 60.50%              |
| 200        | 10     | 78.76%            | 61.00% (epoch 10)   |
| 200        | 20     | 95.03%            | 61.00% (epoch 10)   |

### RNN Results

| Hidden Dim | Epochs Run | Training Accuracy | Validation Accuracy |
|------------|------------|-------------------|---------------------|
| 10         | 4          | 46.18%            | 45.75% (epoch 3)    |
| 50         | 3          | 43.73%            | 43.00% (epoch 2)    |
| 100        | 3          | 41.35%            | 42.00% (epoch 2)    |
| 200        | 3          | 39.96%            | 41.63% (epoch 1)    |

## 4 Analysis

The FFNN significantly outperformed the RNN across all configurations. With 62.63% being the best validation accuracy at hidden_dim=100, epoch 12. The RNN's best validation accuracy was only 45.7% at hidden_dim=10, epoch 3. For the FFNN, increasing the hidden dimension from 10 to 100 improved performance, but the difference between 100 and 200 was negligible (beyond the greatly increased time). The RNN on the other hand surprisingly performed better with smaller hidden dimensions. When changing the number of epochs from 5 to 10 the FFNN validation accuracy improved, but began to plateau after 10 epochs. The RNN typically stopped early after 3-4 epochs, both results seem to imply overfitting was beginning to occur.

The bag-of-words representation seems to be better suited to this specific task of sentiment analysis, the FFNN might have been able to learn from the prevelance of specific words negative/postive reviews. This makes sense as the reviews (being for hotels) might include words such as "disguting" or "filthy" which would serve as strong signifiers of negativity. The sequences are relatively short as well, which might have caused issues for the RNN which typically benefits/learns better from longer sequences 

## 5 Conclusion

The best configuration of the neural networks appeared to be the FNN trained for 12 epochs with a hidden dimension of 100, achieving a validation accuracy of 62.63%. Suggesting that for this sentiment analysis task, a feed-forward network with bag-of-words representation outperforms the recurrent neural network with word embeddings.

The assignment was not too difficult, but took some time to sort out the correct way to implement the forward methods for both the FNN and RNN. The comments suggesting where/what to fill in and the way the output was discarded in the RNN forward method ```_, hidden = self.rnn(inputs)``` left me feeling conflicted about what the intended/implied approach was for this method specifically. Even though it was only a few lines needed, it overall probably took me around 10-15 hours to fully complete with the report.