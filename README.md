# ML-CV
A list of python notebooks that i completed through my courses at UTEP including Deep learning, Machine Learning, and Computer Vision

---

## Foundations

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `arrays_and_numpy.ipynb` | Introduction to NumPy arrays, operations, slicing, broadcasting, and boolean indexing | NumPy, vectorization, array operations |
| `Intro to python.ipynb` | Data analysis with NumPy and Matplotlib on California Housing dataset | NumPy, Matplotlib, data loading, visualization |
| `Accuracy Excercise.ipynb` | Calculating accuracy and precision metrics from predictions | Accuracy, precision, classification metrics |


---

## Classical Machine Learning

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `decision_trees_MLFall2021.ipynb` | Comprehensive decision tree tutorial: ID3 algorithm, entropy, information gain, sklearn | ID3, entropy, information gain, Gini, tree visualization |
| `decision _trees.ipynb` | Custom decision tree implementation finding optimal thresholds | Threshold selection, random splitting |
| `linear_regression_ML_Fall21.ipynb` | Linear regression on GPU running time data using matrix operations and sklearn | Linear regression, least squares, MSE, MAE |
| `ensembles_ML_Fall21.ipynb` | Ensemble methods comparing Decision Trees, Random Forests, and k-NN regression | Bagging, Random Forest, ensemble learning |
| `mnist_knn_fall21.ipynb` | k-NN on MNIST with 3 optimization versions (loops → broadcasting → matrix ops) | k-NN, Euclidean distance, broadcasting, algorithm optimization |
| `mnist_sklearn.ipynb` | MNIST classification using sklearn's KNeighborsClassifier | sklearn, k-NN, accuracy_score, confusion_matrix |
| `sklearn_mlp.ipynb` | Multi-layer perceptron using sklearn for MNIST classification and GPU runtime regression | MLP, neural networks, sklearn |
| `linear_regression_ML_Fall21_solution.ipynb` | Linear regression solution using pseudoinverse and sklearn regressors | Pseudoinverse, least squares, regression |
| `svm_water_start.ipynb` | SVM classification on water potability data with hyperparameter tuning | SVM, kernel methods, MinMaxScaler, C parameter |
| `RandIndex_purity_kmeans.ipynb` | K-Means clustering with Rand Index and purity evaluation metrics | K-Means, clustering, evaluation metrics |
| `performance_metrics.ipynb` | Comprehensive evaluation metrics for ML models | Confusion matrix, precision, recall, F1 |

---

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

## Course Assignments & Exams

### ML Course Exams

| Notebook | Description | Concepts Covered |
|----------|-------------|------------------|
| `Ramos_Estevan_exam1.ipynb` | Feature engineering (removing constant attributes), comparing classifiers (Random Forest, MLP, Logistic Regression) on MNIST | Feature selection, model comparison, accuracy evaluation |
| `Ramos_Estevan_exam2*.ipynb` | Building Dense and CNN models for fiber density image classification (8 classes) | Keras Sequential, Conv2D, Dense layers, categorical cross-entropy |
| `Ramos_Estevan_finalexam.ipynb` | Olivetti Faces classification using PCA + SVM, achieving best accuracy through dimensionality reduction | PCA, SVM, face recognition, hyperparameter tuning |
| `ML_Final_Exam.ipynb` | Intrusion Detection System challenge - classifying network traffic as benign or attack types (DoS, DDoS, BotNet) | Multi-class classification, network security, sklearn classifiers |
| `exam1_solution.ipynb` | Exam 1 solutions and explanations | Reference solutions |
| `exam_2_solution.ipynb` | Exam 2 solutions and explanations | Reference solutions |
| `practice_exam_2.ipynb` | Practice problems for exam preparation | Exam preparation |

### ML Course Labs

| Notebook | Description | Skills Learned |
|----------|-------------|----------------|
| `Ramos_Estevan_lab1.ipynb` | k-NN on MNIST and Fashion-MNIST, implementing confusion matrix from scratch, evaluating with precision/recall/F1 | Custom metric implementation, k-NN, Fashion-MNIST |
| `Ramos_Estevan_Lab2.ipynb` | Water potability classification comparing k-NN, Naive Bayes, and Decision Trees on chemical analysis data | Real-world dataset, classifier comparison, data preprocessing |
| `Ramos_Estevan_lab3.ipynb` | Multi-dataset image classification (CIFAR-10, LFW, CelebA) using sklearn classifiers, Dense networks, and CNNs | End-to-end ML pipeline, multiple architectures, PCA preprocessing |

### Deep Learning Assignments

| Notebook | Description | Skills Learned |
|----------|-------------|----------------|
| `cnn_Ramos_Estevan.ipynb` | CNN architecture for LFW face recognition (62 classes), with Conv2D, MaxPooling, training visualization | CNN design, face recognition, training curves |
| `Dense_Ramos_Estevan.ipynb` | Dense network for LFW face recognition, comparing performance to CNN approach | Dense vs CNN comparison, flattening images |
| `cv_exam2.ipynb` | Fiber density material classification using Dense and CNN models (8-class problem) | Image classification, model architecture design |
| `DL_Project_1.ipynb` | Credit default prediction - data loading and preparation for deep learning | Tabular data, pandas, data preprocessing |
| `regression_HW1.ipynb` | PyTorch regression on Iris dataset with DataLoader, StandardScaler, and custom training loop | PyTorch fundamentals, DataLoader, StandardScaler |
| `Homework 3  foundation and Sequential models.ipynb` | Fine-tuning GPT-2 for text generation using HuggingFace Transformers | GPT-2, Transformers, fine-tuning, NLP |

### Data Mining Assignments

| Notebook | Description | Concepts Learned |
|----------|-------------|------------------|
| `DM_assignment1.ipynb` | Distance metrics implementation: Euclidean, Manhattan, Cosine, Jaccard, Tanimoto; k-NN from scratch | Distance functions, similarity measures, nearest neighbors |
| `DM_assignment2.ipynb` | Clustering evaluation: K-Means, Agglomerative, DBSCAN on dry beans data with Rand Index, Purity, Silhouette Score | Clustering algorithms, cluster evaluation metrics |
| `DMAssigment4.ipynb` | PageRank algorithm implementation, detecting dead ends and spider traps in web graphs | PageRank, link analysis, graph algorithms, matrix operations |

### Projects

| Notebook | Description | Skills Applied |
|----------|-------------|----------------|
| `ML_Final_Project.ipynb` | Cryptocurrency price forecasting using G-Research data, building regression models with RMSE/MAPE metrics | Time series, financial data, regression, feature engineering |

---

## Data & Projects

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `Linear_regression_growth_and_jet_data.ipynb` | Linear regression on growth and jet engine data | Feature engineering, prediction |
| `CNN to enhance CV.ipynb` | Computer vision enhancement using CNNs | Image processing, CV applications |
| `Devcom ML Intro.ipynb` | ML introduction notebook | ML fundamentals |
| `get_jazz_midi_files.ipynb` | Downloading MIDI files for music generation | Data collection, MIDI |
| `messing with offer up.ipynb` | Web scraping OfferUp marketplace listings using pyOfferUp | Web scraping, API, data collection |
| `Copy of DL_Project_1.ipynb` | Copy of deep learning project | Deep learning |
| `Ramos_Estevan_PDE_2022.ipynb` | PDE (Partial Differential Equations) related work | PDE, numerical methods |

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
- A* pathfinding
- Game theory
- Cryptography basics
- Expected value and probability
