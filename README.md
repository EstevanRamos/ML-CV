# ML-CV
A list of python notebooks that i completed through my courses at UTEP including Deep learning, Machine Learning, and Computer Vision

A collection of machine learning, deep learning, and computer science notebooks.

---

## Foundations

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `arrays_and_numpy.ipynb` | Introduction to NumPy arrays, operations, slicing, broadcasting, and boolean indexing | NumPy, vectorization, array operations |
| `Intro to python.ipynb` | Data analysis with NumPy and Matplotlib on California Housing dataset | NumPy, Matplotlib, data loading, visualization |
| `Accuracy Excercise.ipynb` | Calculating accuracy and precision metrics from predictions | Accuracy, precision, classification metrics |

### Intro to python.ipynb - Detailed Breakdown

**Overview:** An introductory Python notebook for working with data using NumPy and Matplotlib on the California Housing dataset.

**Dataset (California Housing - 9 features):**
1. longitude - how far west
2. latitude - how far north
3. housingMedianAge - median age of houses in block
4. totalRooms - total rooms in block
5. totalBedrooms - total bedrooms in block
6. population - people in block
7. households - household count in block
8. medianIncome - median income (tens of thousands USD)
9. medianHouseValue - median house value (USD)

**NumPy Operations Demonstrated:**
- `np.loadtxt()` - loading CSV data
- `np.shape()` - get array dimensions
- Indexing: `data[0,0]`, `data[0]`
- Slicing: `data[:, 7]` (all rows, specific column)

**Matplotlib Visualizations:**
- Line plot of median income
- Scatter plot (`'r*'`) of latitude vs longitude
- Bar chart (vertical) with `ax.bar()`
- Bar chart (horizontal) with `ax.barh()`
- Histograms with subplots for population/household distributions

---

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

### mnist_knn_fall21.ipynb - Detailed Breakdown

**Author:** Olac Fuentes (UTEP)

**Overview:** Building a k-nearest neighbors classifier for MNIST digit recognition, demonstrating algorithm optimization through three progressively faster implementations.

**Dataset:** MNIST - 70,000 grayscale 28x28 images of handwritten digits (0-9)

**Pre-processing Steps:**
1. Reshape images from 28x28 to 784-element vectors
2. Convert from uint8 to float32
3. Scale pixel values to [0,1] range
4. Subsample (every 5th example) for faster development

**k-NN Algorithm:**
```
For each test image:
  1. Compute distance to all training images
  2. Find k nearest neighbors
  3. Predict most common class among neighbors
```

**Three Implementation Versions:**

| Version | Approach | Speed |
|---------|----------|-------|
| V1 | Nested loops (for each test, for each train) | Slowest |
| V2 | Broadcasting - vectorized distance to all training examples | ~20x faster |
| V3 | Full matrix operations using (a-b)² = a² - 2ab + b² | Fastest |

**Key Functions:**
- `distance(p, q)` - Euclidean distance computation
- `most_common(labels)` - Find mode using `scipy.stats.mode()`
- `accuracy(pred, y)` - `np.mean(pred == y)`
- `knn(x_train, y_train, x_test, k)` - Main classifier

**Math Insight (Version 3):**
Distance matrix computed without loops using:
```
d[i,j] = sqrt(sum(A[i]²) - 2*dot(A[i],B[j]) + sum(B[j]²))
```

**Key Takeaways:**
- Vectorization dramatically improves performance
- Broadcasting eliminates explicit loops
- Matrix operations leverage optimized BLAS libraries

### decision_trees_MLFall2021.ipynb - Detailed Breakdown

**Author:** Olac Fuentes (UTEP)

**Overview:** A comprehensive tutorial on decision trees covering theory, the ID3 algorithm, and practical implementation with sklearn.

**Topics Covered:**

**1. Decision Tree Fundamentals**
- Flowchart-like classification model
- Internal nodes = attributes, branches = values, leaves = class labels
- When to use: discrete targets, noisy data, missing values, interpretability needed

**2. ID3 Algorithm**
- Quinlan's algorithm for learning decision trees from data
- Recursive partitioning based on attribute selection
- Pseudocode explanation with step-by-step example

**3. Entropy & Information Gain**

Entropy formula for binary labels:
```
ent(S) = -p(0)*log2(p(0)) - p(1)*log2(p(1))
```

Information Gain:
```
Gain(S,A) = entropy(S) - Σ (|Sv|/|S|) * entropy(Sv)
```

- Entropy = 0 when all examples same class (pure)
- Entropy = 1 when 50/50 split (maximum uncertainty)
- Select attribute with highest information gain

**4. Hand-Worked Example**
- 8 examples with 3 attributes (a0, a1, a2)
- Calculates gain for each attribute
- Shows a1 maximizes gain → chosen as root
- Builds complete tree through recursive splits

**5. sklearn Implementation**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X, y)
tree.plot_tree(model, feature_names=[...])
```

**6. Datasets Used**
| Dataset | Task | Accuracy |
|---------|------|----------|
| Toy example (8 samples) | Binary classification | 100% |
| MNIST | Digit recognition | ~87% |
| DOH (network traffic) | Benign vs malicious | ~99% |

**7. Hyperparameters Explored**
- `criterion`: 'entropy' vs 'gini'
- `max_depth`: Controls tree complexity (1-25 tested)

**Key Takeaways:**
- Information gain guides attribute selection
- Deeper trees can overfit; max_depth controls complexity
- Decision trees are interpretable but may underperform on high-dimensional data (MNIST)
- Work well on tabular data with meaningful features (DOH dataset)

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

### bag_of_words.ipynb - Detailed Breakdown

**Overview:** Text classification tutorial using the Bag-of-Words approach to classify paragraphs from classic literature by their source book.

**Dataset:**
Three books from Project Gutenberg:
| Book | Title | Author |
|------|-------|--------|
| 0 | The Call of the Wild | Jack London |
| 1 | Dracula | Bram Stoker |
| 2 | The Adventures of Sherlock Holmes | Arthur Conan Doyle |

- Paragraphs extracted via web scraping (BeautifulSoup)
- Task: Given a paragraph, predict which book it came from

**Bag-of-Words Pipeline:**
```python
# 1. Build vocabulary from training data
count_vect = CountVectorizer()
count_vect.fit(x_train)

# 2. Transform text to numeric features
x_train_counts = count_vect.transform(x_train)
# word_counts[i,j] = count of word j in document i
```

**Classifiers Compared:**
| Classifier | sklearn Class |
|------------|---------------|
| Naive Bayes | `MultinomialNB()` |
| k-Nearest Neighbors | `KNeighborsClassifier()` |
| Support Vector Machine | `SVC()` |
| Random Forest | `RandomForestClassifier()` |

**CountVectorizer Parameters Explored:**

| Parameter | Effect |
|-----------|--------|
| `min_df=2` | Ignore words appearing in < 2 documents |
| `max_df=0.3` | Ignore words appearing in > 30% of documents |
| `binary=True` | Use presence/absence instead of counts |
| `ngram_range=(1,3)` | Include 1-grams, 2-grams, and 3-grams |

**TF-IDF Transformation:**
```python
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
```
- Term Frequency-Inverse Document Frequency
- Downweights common words, upweights distinctive words

**Key Observations:**
- Filtering rare/common words reduces feature count significantly
- N-grams capture multi-word phrases but increase dimensionality
- Different classifiers perform differently on text data
- TF-IDF often improves over raw counts

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
