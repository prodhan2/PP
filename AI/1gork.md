### Assignment 1: Manually Draw a Fully Connected Feed-forward Neural Network (FCFNN)

Since this is a manual drawing task, I can't physically draw it here, but I can provide a textual representation using ASCII art to visualize the network structure. The network has:
- Input layer: 8 neurons
- Hidden layer 1: 4 neurons
- Hidden layer 2: 8 neurons
- Hidden layer 3: 4 neurons
- Output layer: 10 neurons

Each neuron in one layer is fully connected to every neuron in the next layer.

ASCII Representation:

```
Input Layer (8 neurons):
O O O O O O O O

          | (full connections to all in Hidden 1)
          
Hidden Layer 1 (4 neurons):
O O O O

          | (full connections to all in Hidden 2)
          
Hidden Layer 2 (8 neurons):
O O O O O O O O

          | (full connections to all in Hidden 3)
          
Hidden Layer 3 (4 neurons):
O O O O

          | (full connections to all in Output)
          
Output Layer (10 neurons):
O O O O O O O O O O
```

To draw this manually:
- Use paper/graph paper.
- Represent neurons as circles.
- Draw lines connecting every neuron from one layer to the next.
- Label layers and neuron counts.

### Assignment 2: Write a Report in PDF Format Using LaTeX After Drawing a FCFNN and Implementing It Using TensorFlow.Keras

For the report, I'll provide the LaTeX source code you can compile to PDF (e.g., using Overleaf or pdflatex). I chose a custom FCFNN: Input 5 neurons, Hidden layers: 6, 12, 6 neurons, Output 3 neurons.

#### LaTeX Report Source Code
```latex
\documentclass{article}
\usepackage{graphicx} % For including images
\usepackage{listings} % For code listings
\usepackage[margin=1in]{geometry}

\title{FCFNN Design and Implementation Report}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This report describes a Fully Connected Feed-forward Neural Network (FCFNN) with 5 input neurons, three hidden layers (6, 12, 6 neurons), and 3 output neurons. The network was drawn manually and implemented using TensorFlow.Keras.

\section{Network Diagram}
% Assume you save your manual drawing as 'fcfnn_diagram.png' and include it here.
\includegraphics[width=\textwidth]{fcfnn_diagram.png}

Description: The input layer has 5 neurons, fully connected to the first hidden layer with 6 neurons. This connects to a second hidden layer with 12 neurons, then to a third with 6, and finally to the output with 3 neurons.

\section{Implementation in TensorFlow.Keras}
The network was implemented as follows:

\lstset{language=Python, basicstyle=\small\ttfamily, breaklines=true}
\begin{lstlisting}
# Full code with comments
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential([
    # Input layer is implicit in Keras, but we specify input shape
    layers.Dense(6, activation='relu', input_shape=(5,)),  # Hidden Layer 1: 6 neurons, ReLU activation
    layers.Dense(12, activation='relu'),                   # Hidden Layer 2: 12 neurons, ReLU
    layers.Dense(6, activation='relu'),                    # Hidden Layer 3: 6 neurons, ReLU
    layers.Dense(3, activation='softmax')                  # Output Layer: 3 neurons, Softmax for classification
])

# Compile the model
model.compile(optimizer='adam',                            # Optimizer: Adam
              loss='sparse_categorical_crossentropy',      # Loss: For multi-class classification
              metrics=['accuracy'])                        # Metric: Accuracy

# Summary of the model
model.summary()
\end{lstlisting}

\section{Conclusion}
The FCFNN was successfully designed and implemented. It can be trained on appropriate data.

\end{document}
```

Compile this LaTeX to PDF. Replace `\includegraphics` with your actual drawing image.

#### Full Python Code with Comments (For Implementation)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model architecture
# Sequential model for feed-forward network
model = models.Sequential([
    # First hidden layer: 6 neurons, ReLU activation, input shape matches 5 features
    layers.Dense(6, activation='relu', input_shape=(5,)),
    # Second hidden layer: 12 neurons, ReLU
    layers.Dense(12, activation='relu'),
    # Third hidden layer: 6 neurons, ReLU
    layers.Dense(6, activation='relu'),
    # Output layer: 3 neurons, softmax for multi-class output
    layers.Dense(3, activation='softmax')
])

# Compile the model with optimizer, loss, and metrics
model.compile(optimizer='adam',  # Adaptive Moment Estimation optimizer
              loss='sparse_categorical_crossentropy',  # Suitable for integer-labeled classification
              metrics=['accuracy'])  # Track accuracy during training

# Print model summary to see layers, parameters, etc.
model.summary()
```

### Assignment 3: Write a Report in PDF Format Using LaTeX After Building FCFNNs for Equations

LaTeX Report Source (compile to PDF):

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{listings}
\usepackage[margin=1in]{geometry}

\title{FCFNN for Equation Solving Report}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
FCFNNs were built for y = 5x + 10, y = 3x² + 5x + 10, y = 4x³ + 3x² + 5x + 10. Data was prepared, models trained/tested, and effects discussed.

\section{Data Preparation}
Generated 1000 samples for x in [-10, 10]. Split: 70% train, 15% val, 15% test.

\section{Models and Training}
For linear: 1 hidden layer (10 neurons). For quadratic: 2 hidden (10,10). For cubic: 3 hidden (10,20,10).

\section{Plots}
% Include matplotlib plots as images
\includegraphics[width=0.5\textwidth]{linear_plot.png}
\includegraphics[width=0.5\textwidth]{quadratic_plot.png}
\includegraphics[width=0.5\textwidth]{cubic_plot.png}

\section{Discussion}
Higher power requires more layers/neurons and data to capture non-linearity.

\section{Code}
\lstset{language=Python}
\begin{lstlisting}
# Full code here (abbreviated)
\end{lstlisting}

\end{document}
```

#### Full Python Code with Comments (For All Equations)
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to generate data for an equation
def generate_data(equation, x_range=(-10, 10), num_samples=1000):
    x = np.linspace(x_range[0], x_range[1], num_samples)
    if equation == 'linear':
        y = 5 * x + 10  # y = 5x + 10
    elif equation == 'quadratic':
        y = 3 * x**2 + 5 * x + 10  # y = 3x² + 5x + 10
    elif equation == 'cubic':
        y = 4 * x**3 + 3 * x**2 + 5 * x + 10  # y = 4x³ + 3x² + 5x + 10
    return x.reshape(-1, 1), y.reshape(-1, 1)  # Reshape for TF input

# Split data into train/val/test
def split_data(x, y, train_ratio=0.7, val_ratio=0.15):
    total = len(x)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return (x[:train_end], y[:train_end]), (x[train_end:val_end], y[train_end:val_end]), (x[val_end:], y[val_end:])

# Build and train FCFNN model
def build_train_model(input_shape, hidden_layers, equation):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))  # Input layer
    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation='relu'))  # Hidden layers with ReLU
    model.add(layers.Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Mean Squared Error for regression
    return model

# Plot original vs predicted
def plot_predictions(x_test, y_test, y_pred, title):
    plt.figure()
    plt.plot(x_test, y_test, label='Original y')
    plt.plot(x_test, y_pred, label='Predicted y')
    plt.title(title)
    plt.legend()
    plt.show()  # Or savefig for report

# Main execution for each equation
equations = ['linear', 'quadratic', 'cubic']
hidden_configs = {'linear': [10], 'quadratic': [10, 10], 'cubic': [10, 20, 10]}  # Adjust based on power

for eq in equations:
    x, y = generate_data(eq)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)
    
    model = build_train_model((1,), hidden_configs[eq], eq)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, verbose=0)  # Train
    
    y_pred = model.predict(x_test)  # Test
    print(f'{eq.capitalize()} MSE: {tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy().mean()}')  # Evaluate
    
    plot_predictions(x_test, y_test, y_pred, f'{eq.capitalize()} Equation')

# Discussion: Higher power (e.g., cubic) needs more layers/neurons and data to avoid underfitting due to increased non-linearity.
```

Save plots as images for the LaTeX report. Higher power increases model complexity and data needs for accurate fitting.

### Assignment 4: Write a Report in PDF Format Using LaTeX After Building FCFNN Classifier

LaTeX similar to above, include code and results.

#### Full Python Code with Comments (For Fashion MNIST, MNIST, CIFAR-10)
I chose 3 hidden layers: 256, 128, 64 neurons.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
from tensorflow.keras.utils import to_categorical

# Function to load and preprocess dataset
def load_preprocess_dataset(dataset_name):
    if dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize and flatten for FCFNN (CNN would keep shape)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    if len(x_train.shape) > 2:  # Flatten for dense layers
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    y_train = to_categorical(y_train)  # One-hot encode
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

# Build FCFNN classifier
def build_fcfnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),  # Hidden 1
        layers.Dense(128, activation='relu'),                          # Hidden 2
        layers.Dense(64, activation='relu'),                           # Hidden 3
        layers.Dense(num_classes, activation='softmax')                # Output
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Datasets
datasets = ['fashion_mnist', 'mnist', 'cifar10']

for ds in datasets:
    (x_train, y_train), (x_test, y_test) = load_preprocess_dataset(ds)
    input_shape = (x_train.shape[1],)
    num_classes = y_train.shape[1]
    
    model = build_fcfnn(input_shape, num_classes)
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=128, verbose=1)  # Train
    
    test_loss, test_acc = model.evaluate(x_test, y_test)  # Test
    print(f'{ds} Test Accuracy: {test_acc}')
```

Include accuracy in report. FCFNN works well for MNIST/Fashion but poorly for CIFAR-10 due to no spatial awareness.

### Assignment 5: Write a Report in PDF Format Using LaTeX After Building CNN Classifier

Similar LaTeX structure.

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess (keep image shape for CNN)
def load_preprocess_dataset(dataset_name):
    if dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train[..., tf.newaxis]  # Add channel dim
        x_test = x_test[..., tf.newaxis]
    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Already RGB
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

# Build CNN for 10-class classification
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),  # Conv1
        layers.MaxPooling2D((2,2)),                                           # Pool1
        layers.Conv2D(64, (3,3), activation='relu'),                          # Conv2
        layers.MaxPooling2D((2,2)),                                           # Pool2
        layers.Flatten(),                                                     # Flatten
        layers.Dense(128, activation='relu'),                                 # Dense
        layers.Dense(num_classes, activation='softmax')                       # Output
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Datasets
datasets = ['fashion_mnist', 'mnist', 'cifar10']

for ds in datasets:
    (x_train, y_train), (x_test, y_test) = load_preprocess_dataset(ds)
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1]
    
    model = build_cnn(input_shape, num_classes)
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64, verbose=1)
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'{ds} Test Accuracy: {test_acc}')
```

CNN performs better than FCFNN on all, especially CIFAR-10.

### Assignment 6: Write a Report in PDF Format Using LaTeX After Preparing Handwritten Digit Dataset and Retraining FCFNN

For your own dataset, you'll need to collect images (e.g., 100 handwritten digits 0-9, scan/photograph, resize to 28x28 grayscale). Assume you have them in a folder. Use code to load.

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2  # For loading your images; install if needed, but assuming available
import os

# Load MNIST
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

# Load your custom dataset (assume folder 'custom_digits' with images named 'digit_0_1.jpg' etc.)
def load_custom_dataset(folder_path, num_samples_per_class=10):
    x_custom = []
    y_custom = []
    for digit in range(10):
        for i in range(num_samples_per_class):
            img_path = os.path.join(folder_path, f'digit_{digit}_{i}.jpg')  # Adjust naming
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
            img = cv2.resize(img, (28, 28))  # Resize to match MNIST
            x_custom.append(img)
            y_custom.append(digit)
    x_custom = np.array(x_custom).astype('float32') / 255.0
    y_custom = np.array(y_custom)
    return x_custom, y_custom

# Split custom into train/test (80/20)
custom_folder = 'path/to/custom_digits'  # Replace with your path
x_custom, y_custom = load_custom_dataset(custom_folder, 10)  # 100 samples
split = int(0.8 * len(x_custom))
x_custom_train, y_custom_train = x_custom[:split], y_custom[:split]
x_custom_test, y_custom_test = x_custom[split:], y_custom[split:]

# Combine with MNIST train
x_train_combined = np.concatenate([mnist_x_train, x_custom_train])
y_train_combined = np.concatenate([mnist_y_train, y_custom_train])

# Preprocess
x_train_combined = x_train_combined.reshape(-1, 784).astype('float32') / 255.0  # Flatten if needed, but already /255 for MNIST
mnist_x_test = mnist_x_test.reshape(-1, 784).astype('float32') / 255.0
x_custom_test = x_custom_test.reshape(-1, 784).astype('float32') / 255.0
y_train_combined = to_categorical(y_train_combined, 10)
mnist_y_test = to_categorical(mnist_y_test, 10)
y_custom_test = to_categorical(y_custom_test, 10)

# Build FCFNN
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Retrain on combined
model.fit(x_train_combined, y_train_combined, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate on combined test (MNIST test + custom test)
x_test_combined = np.concatenate([mnist_x_test, x_custom_test])
y_test_combined = np.concatenate([mnist_y_test, y_custom_test])
test_loss, test_acc = model.evaluate(x_test_combined, y_test_combined)
print(f'Combined Test Accuracy: {test_acc}')
```

Report: Accuracy may drop slightly due to variability in custom data.

### Assignment 7: Write a Report in PDF Format Using LaTeX After Training CNN with Own Images

Collect images (e.g., 10 classes, 100 per class, from phone). Assume folder structure like 'own_dataset/class1/img1.jpg'.

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Assume dataset in 'own_dataset/train' and 'own_dataset/test' with subfolders for classes
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory('own_dataset/train', target_size=(32,32), batch_size=32, class_mode='categorical', subset='training')
val_generator = train_datagen.flow_from_directory('own_dataset/train', target_size=(32,32), batch_size=32, class_mode='categorical', subset='validation')
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory('own_dataset/test', target_size=(32,32), batch_size=32, class_mode='categorical')

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assume 10 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and time
start_time = time.time()
history = model.fit(train_generator, epochs=20, validation_data=val_generator)
training_time = time.time() - start_time
print(f'Total Training Time: {training_time} seconds')

# Test and time per sample
start_time = time.time()
test_loss, test_acc = model.evaluate(test_generator)
testing_time = (time.time() - start_time) / len(test_generator)
print(f'Test Accuracy: {test_acc}, Testing Time per Sample: {testing_time} seconds')

# Model size
print(f'Number of Parameters: {model.count_params()}')

# Plots: epoch vs acc/loss, data size vs perf (run with subsets), etc.
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.show()
```

Report: Discuss training time (~minutes), test time (~ms/sample), more data improves perf, more epochs up to point, larger model better but slower.

### Assignment 8: Build a CNN Based Classifier Having Architecture Similar to Classical VGG16

VGG16 has 13 conv + 3 dense layers. Here's a simplified version.

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Build VGG16-like CNN (simplified for 10 classes, input 224x224x3)
def build_vgg16_like(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2), strides=2),
        
        # Block 2
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2), strides=2),
        
        # Block 3
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2), strides=2),
        
        # Block 4
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2), strides=2),
        
        # Block 5
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2), strides=2),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_vgg16_like()
model.summary()  # Shows ~138M params like original
```

Train on your dataset as in previous assignments.

### Assignment 9: Write a Report on How Feature Maps Look When Passing Favorite Image Through Favorite Pre-trained CNN Classifiers

Choose favorites: VGG16, ResNet50, InceptionV3. Favorite image: e.g., a cat photo (download one).

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained models (include_top=False for features)
models_dict = {
    'VGG16': VGG16(weights='imagenet', include_top=False),
    'ResNet50': ResNet50(weights='imagenet', include_top=False),
    'InceptionV3': InceptionV3(weights='imagenet', include_top=False)
}

# Load image
img_path = 'path/to/your_favorite_image.jpg'  # e.g., cat.jpg, resize to 224x224
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)  # Preprocess for VGG, adjust for others if needed

# Function to get feature maps from a layer
def get_feature_maps(model, layer_name, input_img):
    layer_output = model.get_layer(layer_name).output
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    return activation_model.predict(input_img)

# Visualize feature maps
def plot_feature_maps(feature_maps, model_name, layer_name):
    num_maps = feature_maps.shape[-1]
    fig, axs = plt.subplots(1, min(8, num_maps))  # Show first 8
    for i in range(min(8, num_maps)):
        axs[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
        axs[i].axis('off')
    plt.suptitle(f'{model_name} - {layer_name}')
    plt.show()

# For each model, pick conv layers and visualize
for model_name, model in models_dict.items():
    # Example layers: early, mid, late
    layers_to_vis = ['block1_conv1', 'block3_conv1', 'block5_conv1'] if model_name == 'VGG16' else \
                    ['conv1_conv', 'conv3_block1_1_conv', 'conv5_block1_1_conv'] if model_name == 'ResNet50' else \
                    ['conv2d', 'mixed5', 'mixed10']
    for layer in layers_to_vis:
        if hasattr(model, 'get_layer') and model.get_layer(layer) is not None:  # Check layer exists
            fm = get_feature_maps(model, layer, x)
            plot_feature_maps(fm, model_name, layer)
```

Report: Early layers show edges/textures, mid show parts (e.g., eyes), late show abstract concepts. Save plots for PDF.

### Assignment 10: Write a Report in PDF Format Using LaTeX After Training Binary Classifier with Transfer Learning on VGG16

Assume binary dataset (e.g., cats vs dogs from TF datasets).

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10  # Example, modify for binary (e.g., classes 0 vs 1)

# Load binary data (example: CIFAR10 classes 0=airplane, 1=auto as binary)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Make binary: keep only classes 0 and 1
train_mask = (y_train == 0) | (y_train == 1)
test_mask = (y_test == 0) | (y_test == 1)
x_train, y_train = x_train[train_mask[:,0]], y_train[train_mask[:,0]]
x_test, y_test = x_test[test_mask[:,0]], y_test[test_mask[:,0]]
y_train = (y_train == 1).astype(int)  # Binary 0/1
y_test = (y_test == 1).astype(int)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))  # CIFAR is 32x32

# Function for transfer learning
def build_transfer_model(fine_tune_layers=None):
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    if fine_tune_layers == 'whole':
        base_model.trainable = True  # Fine-tune whole
    elif fine_tune_layers == 'partial':
        base_model.trainable = True
        for layer in base_model.layers[:10]:  # Freeze first 10, fine-tune rest
            layer.trainable = False
    else:
        base_model.trainable = False  # Transfer learning only
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate
for mode in ['transfer_only', 'partial', 'whole']:
    model = build_transfer_model(mode if mode != 'transfer_only' else None)
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'{mode} Test Acc: {test_acc}')
```

Report: Fine-tuning whole improves acc but risks overfitting; partial balances.

### Assignment 11: Discuss Feature Extraction Power Before/After Transfer Learning Using PCA and t-SNE

Use MNIST for transfer (though ImageNet pre-trained on RGB, convert MNIST to RGB).

#### Full Python Code with Comments
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load MNIST, convert to RGB for VGG
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.repeat(x_test[..., np.newaxis], 3, -1)  # Grayscale to RGB
x_test = tf.image.resize(x_test, (224, 224)).numpy() / 255.0  # Resize and normalize

# Pre-trained VGG
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')  # Global avg pooling for features

# Extract features before transfer
features_before = base_model.predict(x_test[:500])  # Subset for speed

# Transfer learn on MNIST (train a classifier head)
model = tf.keras.Sequential([base_model, layers.Dense(10, activation='softmax')])
base_model.trainable = False  # Transfer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train[..., np.newaxis], y_train, epochs=5)  # Train head (simplified, x_train prep similar)

# Extract after (use the base after training, but since frozen, same; for fine-tune, unfreeze)
# For after, assume fine-tuned (unfreeze for demo)
base_model.trainable = True
model.fit(x_train[..., np.newaxis], y_train, epochs=5)  # Fine-tune
features_after = base_model.predict(x_test[:500])

# Dimension reduction
def plot_reduction(features, labels, title, method='pca'):
    if method == 'pca':
        reduced = PCA(n_components=2).fit_transform(features)
    else:
        reduced = TSNE(n_components=2).fit_transform(features)
    plt.figure()
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels[:len(reduced)], cmap='jet')
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

# Plots
for method in ['pca', 'tsne']:
    plot_reduction(features_before, y_test[:500], f'Before Transfer - {method.upper()}', method)
    plot_reduction(features_after, y_test[:500], f'After Transfer - {method.upper()}', method)
```

Discussion: Before: Features clustered loosely (ImageNet for objects, not digits). After: Tighter clusters, better separation as adapted to digits.

### Assignment 12: Write a Report by Discussing the Effect of Different Data Augmentation Techniques on CNN Classifiers

Use ImageDataGenerator for aug (rotation, flip, etc.).

#### Full Python Code with Comments (Example on CIFAR-10)
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build simple CNN
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# No aug
model_no_aug = build_cnn()
history_no = model_no_aug.fit(x_train, y_train, epochs=10, validation_split=0.2)

# With aug: rotation, flip, shift
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
model_aug = build_cnn()
history_aug = model_aug.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))

# Compare
print(f'No Aug Val Acc: {max(history_no.history["val_accuracy"])}')
print(f'With Aug Val Acc: {max(history_aug.history["val_accuracy"])}')
```

Report: Augmentation (rotation, flip) improves generalization, reduces overfitting, higher val acc by simulating variations.

### Assignment 13: Show the Effect of Dropout Layer, Data Augmentation on Overfitting

Extend above code.

#### Full Python Code with Comments
```python
# Build CNN with dropout
def build_cnn_with_dropout():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout 50%
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train without dropout/aug
model_base = build_cnn()
history_base = model_base.fit(x_train, y_train, epochs=20, validation_split=0.2)

# With dropout
model_dropout = build_cnn_with_dropout()
history_dropout = model_dropout.fit(x_train, y_train, epochs=20, validation_split=0.2)

# With aug + dropout
model_aug_dropout = build_cnn_with_dropout()
history_aug_dropout = model_aug_dropout.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=20, validation_data=(x_test, y_test))

# Plot overfitting: train vs val acc
def plot_overfitting(history, title):
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(title)
    plt.legend()
    plt.show()

plot_overfitting(history_base, 'Base')
plot_overfitting(history_dropout, 'With Dropout')
plot_overfitting(history_aug_dropout, 'With Aug + Dropout')
```

Effect: Base shows gap (overfit). Dropout reduces gap. Aug + dropout minimizes, better generalization.

### Assignment 14: Write a Report by Discussing the Effect of Activation Functions and Loss Functions

#### Full Python Code with Comments (On MNIST for Simplicity)
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Function to build model with variable activation/loss
def build_model(activation='relu', loss='categorical_crossentropy'):
    model = models.Sequential([
        layers.Dense(128, activation=activation, input_shape=(784,)),
        layers.Dense(64, activation=activation),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

# Test activations
activations = ['relu', 'sigmoid', 'tanh']
for act in activations:
    model = build_model(act)
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    test_acc = model.evaluate(x_test, y_test)[1]
    print(f'{act} Acc: {test_acc}')

# Test losses (for classification)
losses = ['categorical_crossentropy', 'sparse_categorical_crossentropy']  # Sparse if y not one-hot
for ls in losses:
    model = build_model(loss=ls)
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    test_acc = model.evaluate(x_test, y_test)[1]
    print(f'{ls} Acc: {test_acc}')
```

Report: ReLU > Tanh > Sigmoid (avoids vanishing gradients). Crossentropy losses similar; sparse convenient if labels integer.

### Assignment 15: Write a Report Describing How Callback Functions Can Make Training Better

Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard.

#### Example Code with Comments
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Build model (use any from above)
model = build_cnn()  # Example

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),  # Stop if no improvement
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),  # Save best
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),  # Reduce LR on plateau
    TensorBoard(log_dir='logs')  # For visualization
]

# Train with callbacks
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=callbacks)
```

Report: EarlyStopping prevents overfitting. Checkpoint saves best model. ReduceLR improves convergence. TensorBoard monitors.

### Assignment 16: Write a Report Describing How Monitoring Performance Curves Improves Hyperparameter Tuning

Plot acc/loss curves for train/val.

#### Code for Monitoring
```python
# After training (e.g., from any history)
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()
```

Report: If val loss rises while train falls: overfit → add dropout/aug, reduce epochs. Plateau: reduce LR. Use to tune batch_size, epochs, layers via trial/error or grid search.
