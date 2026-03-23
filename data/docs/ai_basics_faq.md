# AI & Machine Learning — Basics FAQ

## What is Machine Learning?
Machine Learning (ML) is a subfield of Artificial Intelligence where systems learn patterns from data without being explicitly programmed. Instead of writing rules manually, we show the model many examples and let it discover the underlying patterns.

## What is the difference between ML, DL, and AI?
Artificial Intelligence (AI) is the broad field of building systems that mimic human intelligence. Machine Learning (ML) is a subset of AI focused on learning from data. Deep Learning (DL) is a subset of ML that uses neural networks with many layers (hence "deep") to learn hierarchical representations from data.

## What is supervised learning?
In supervised learning, the model is trained on labeled data — every training example has a known input and output. The model learns a mapping from input to output. Common tasks include classification (predicting a category) and regression (predicting a number).

## What is unsupervised learning?
Unsupervised learning uses unlabeled data. The model tries to find structure on its own — for example, clustering similar data points together (K-Means), or compressing data into a lower-dimensional space (PCA, Autoencoders).

## What is a neural network?
A neural network is a computational model inspired by the human brain. It consists of layers of neurons (nodes), where each neuron applies a weighted sum followed by a non-linear activation function. By stacking many layers and training with backpropagation, neural networks learn complex functions.

## What is overfitting?
Overfitting occurs when a model learns the training data too well — including noise — and performs poorly on new, unseen data. Common solutions include: using more training data, applying regularisation (L1/L2), using dropout, or simplifying the model.

## What is a training/validation/test split?
The dataset is divided into three parts:
- **Training set**: used to train the model.
- **Validation set**: used to tune hyperparameters and detect overfitting.
- **Test set**: held out until the very end to evaluate final model performance.

## What is gradient descent?
Gradient descent is the optimisation algorithm used to train ML models. It iteratively adjusts model parameters in the direction that reduces the loss function. Variants include Stochastic Gradient Descent (SGD), Mini-batch GD, Adam, and RMSProp.
