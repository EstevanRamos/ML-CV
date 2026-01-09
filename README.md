# ML-CV
A list of python notebooks that i completed through my courses at UTEP including Deep learning, Machine Learning, and Computer Vision


---

## Foundations

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `arrays_and_numpy.ipynb` | Introduction to NumPy arrays, operations, slicing, broadcasting, and boolean indexing | NumPy, vectorization, array operations |
| `Intro to python.ipynb` | Data analysis with NumPy and Matplotlib on California Housing dataset | NumPy, Matplotlib, data loading, visualization |
| `Accuracy Excercise.ipynb` | Calculating accuracy and precision metrics from predictions | Accuracy, precision, classification metrics |

## Classical Machine Learning

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `decision_trees_MLFall2021.ipynb` | Comprehensive decision tree tutorial: ID3 algorithm, entropy, information gain, sklearn | ID3, entropy, information gain, Gini, tree visualization |
| `decision _trees.ipynb` | Custom decision tree implementation finding optimal thresholds | Threshold selection, random splitting |
| `linear_regression_ML_Fall21.ipynb` | Linear regression on GPU running time data using matrix operations and sklearn | Linear regression, least squares, MSE, MAE |
| `ensembles_ML_Fall21.ipynb` | Ensemble methods comparing Decision Trees, Random Forests, and k-NN regression | Bagging, Random Forest, ensemble learning |
| `mnist_knn_fall21.ipynb` | k-NN on MNIST with 3 optimization versions (loops → broadcasting → matrix ops) | k-NN, Euclidean distance, broadcasting, algorithm optimization |
| `sklearn_mlp.ipynb` | Multi-layer perceptron using sklearn for MNIST classification and GPU runtime regression | MLP, neural networks, sklearn |
| `svm_water_start.ipynb` | SVM classification on water potability data with hyperparameter tuning | SVM, kernel methods, MinMaxScaler, C parameter |
| `RandIndex_purity_kmeans.ipynb` | K-Means clustering with Rand Index and purity evaluation metrics | K-Means, clustering, evaluation metrics |
| `performance_metrics.ipynb` | Comprehensive evaluation metrics for ML models | Confusion matrix, precision, recall, F1 |


## Deep Learning - Dense Networks

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `keras_dense_mnist_cifar.ipynb` | Dense neural networks in Keras on MNIST, Fashion-MNIST, and CIFAR-10 | Dense layers, softmax, cross-entropy |
| `Dense Neural Network Excercise1.ipynb` | Experimenting with dense network hyperparameters (layers, learning rate, activation functions) on CIFAR-10 | Hyperparameter tuning, overfitting, activation functions |
| `Neural Net from scratch.ipynb` | Building a neural network from scratch with forward pass and backpropagation concepts | Forward propagation, backpropagation, gradient descent |

---

## Deep Learning - Convolutional Neural Networks

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `cifar_simple_cnn.ipynb` | CNN for CIFAR-10 with dropout, batch normalization, and data augmentation | Conv2D, pooling, dropout, batch norm, regularization |
| `mnistt_simple_cnn.ipynb` | Simple CNN achieving ~99% accuracy on MNIST | CNN architecture, training curves |
| `Building Convolutions and pooling .ipynb` | Manual implementation of convolution filters and max pooling | Convolution operation, edge detection, pooling |
| `keras_dense_cnn_MLFall21.ipynb` | Comparing dense and CNN architectures | CNN vs dense, architecture design |

---

## Deep Learning - Advanced Topics

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `sklearn_faces.ipynb` | Face recognition using PCA (eigenfaces), SVM, dense networks, and CNN on LFW dataset | PCA, eigenfaces, dimensionality reduction, face recognition |
| `functional_api_v2.ipynb` | Keras Functional API for complex model architectures | Multi-input/output models, model sharing |
| `adversarial_cifar.ipynb` | Generating adversarial examples to fool trained CNN classifiers | Adversarial attacks, gradient-based perturbations, model robustness |
| `GAN Homework.ipynb` | Generative Adversarial Network implementation in PyTorch | GAN, generator, discriminator, synthetic data |

---

## Natural Language Processing

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `bag_of_words.ipynb` | Text classification on Gutenberg books using CountVectorizer, TF-IDF, and multiple classifiers | Bag of words, TF-IDF, n-grams, text vectorization |
| `pretrained_embeddings_and_word_analogies.ipynb` | Using GloVe embeddings for word similarity and solving word analogies | Word embeddings, GloVe, cosine similarity, word analogies |
| `learning_word_embeddings.ipynb` | Training word embeddings from scratch for text classification (books classification) | Embedding layer, 1D CNN, text classification |

---

## Algorithms & Simulations

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `Monte Carlo.ipynb` | Monte Carlo Tree Search for Connect Four game | MCTS, UCT, game trees, simulation |
| `Poker_sims.ipynb` | Monte Carlo Tree Search applied to poker-style card game (AKQ game) | Game theory, MCTS, decision making |
| `bit_wise_file_Encryption.ipynb` | Simple XOR-based file encryption | XOR cipher, bitwise operations, file I/O |

---

## Course Assignments & Exams

| Notebook | Description |
|----------|-------------|
| `Ramos_Estevan_exam*.ipynb` | Various ML course exam submissions |
| `Ramos_Estevan_practice_exam*.ipynb` | Practice exam work |
| `Ramos_Estevan_lab*.ipynb` | Lab assignments |
| `DM_assignment*.ipynb` | Data Mining course assignments |
| `ML_Final_Project.ipynb` | Final ML project work |
| `DL_Project_1.ipynb` | Deep learning project on credit default prediction |
| `regression_HW1.ipynb` | Regression homework |
| `Homework 3  foundation and Sequential models.ipynb` | Keras Sequential model homework |

---

## Data & Projects

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `Linear_regression_growth_and_jet_data.ipynb` | Linear regression on growth and jet engine data | Feature engineering, prediction |
| `CNN to enhance CV.ipynb` | Computer vision enhancement using CNNs | Image processing, CV applications |
| `Devcom ML Intro.ipynb` | ML introduction notebook | ML fundamentals |
| `get_jazz_midi_files.ipynb` | Downloading MIDI files for music generation | Data collection, MIDI |

---

## Core ML/CS Concepts Covered

**Machine Learning Fundamentals**
- Supervised learning (classification, regression)
- Unsupervised learning (clustering)
- Model evaluation (accuracy, precision, recall, confusion matrix)
- Train/test split, cross-validation

**Deep Learning**
- Neural network architectures (Dense, CNN)
- Activation functions (ReLU, sigmoid, softmax)
- Loss functions (cross-entropy, MSE)
- Regularization (dropout, batch normalization, L2)
- Optimization (Adam, SGD)

**Computer Vision**
- Convolution and pooling operations
- Image classification (MNIST, CIFAR-10, faces)
- Data augmentation
- Adversarial examples

**Natural Language Processing**
- Word embeddings (learned and pretrained)
- Text classification
- 1D convolutions for text

**Algorithms**
- Monte Carlo Tree Search
- Game theory
- Cryptography basics
