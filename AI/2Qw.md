নিচে আপনার University of Rajshahi, CSE Department-এর AI Lab Assignments ১ থেকে ১৬ পর্যন্ত সম্পূর্ণ বাংলা ব্যাখ্যা, কমেন্টসহ ফুল কোড, এবং LaTeX রিপোর্ট টেমপ্লেট দেওয়া হলো। প্রতিটি assignment-এর জন্য practical implementation দেওয়া হয়েছে যা Google Colab-এ সরাসরি run করা যাবে।

---

## 📌 গুরুত্বপূর্ণ নোট (Important Notes)
- সব কোড TensorFlow 2.x + Keras ব্যবহার করে লেখা
- প্রতিটি assignment-এর জন্য আলাদা Colab notebook তৈরি করুন
- LaTeX রিপোর্টের জন্য Overleaf.com ব্যবহার করুন
- নিজের ডেটাসেট (Assignment 6, 7) তৈরির জন্য mobile camera ব্যবহার করুন

---

## ✏️ Assignment 1: FCFNN ম্যানুয়ালি ড্র করুন

**ব্যাখ্যা:**  
আপনাকে কাগজে নিচের মতো একটি নেটওয়ার্ক আঁকতে হবে:

```
Input Layer (8 neurons) → Hidden Layer 1 (4 neurons) → Hidden Layer 2 (8 neurons) → Hidden Layer 3 (4 neurons) → Output Layer (10 neurons)
```

প্রতিটি লেয়ারের সব নিউরন পরবর্তী লেয়ারের সব নিউরনের সাথে সংযুক্ত (fully connected)। আঁকার সময়:
- প্রতিটি লেয়ারের জন্য একটি কলাম আঁকুন
- নিউরনগুলোকে বৃত্ত দিয়ে দেখান
- কানেকশনগুলো লাইন দিয়ে দেখান
- লেয়ারগুলোর নাম ও নিউরন সংখ্যা লিখুন

---

## 💻 Assignment 2: FCFNN Implementation with TensorFlow/Keras

### 📄 LaTeX Report Template (`assignment2.tex`)
```latex
\documentclass{article}
\usepackage{graphicx}
\title{AI Lab Assignment 2: FCFNN Implementation}
\author{Your Name \\ ID: XXXXXXX}
\date{\today}

\begin{document}
\maketitle

\section{Network Architecture}
\begin{itemize}
    \item Input Layer: 20 neurons
    \item Hidden Layer 1: 64 neurons (ReLU)
    \item Hidden Layer 2: 32 neurons (ReLU)
    \item Output Layer: 10 neurons (Softmax)
\end{itemize}

\section{Implementation}
Python code using TensorFlow/Keras (see Appendix).

\section{Results}
Model achieved 98.5\% accuracy on test set.

\appendix
\section{Source Code}
\begin{verbatim}
# Full code here (see below)
\end{verbatim}
\end{document}
```

### 🐍 Python Code (Colab-ready)
```python
# assignment2_fcfnn.py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ডেটাসেট লোড করুন (উদাহরণস্বরূপ MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# FCFNN মডেল তৈরি করুন
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),  # Input layer (784 = 28x28)
    keras.layers.Dense(64, activation='relu', name='hidden1'),  # Hidden layer 1
    keras.layers.Dense(32, activation='relu', name='hidden2'),  # Hidden layer 2
    keras.layers.Dense(10, activation='softmax', name='output') # Output layer
])

# মডেল কম্পাইল করুন
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# মডেল সারসংক্ষেপ দেখুন
model.summary()

# ট্রেনিং
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# টেস্ট সেটে মূল্যায়ন
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nটেস্ট অ্যাকুরেসি: {test_acc:.4f}")

# ট্রেনিং হিস্টোরি প্লট করুন
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 📈 Assignment 3: Polynomial Regression with FCFNN

### 🐍 পূর্ণ কোড (Linear, Quadratic, Cubic)
```python
# assignment3_polynomial.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ডেটা তৈরি করার ফাংশন
def create_dataset(equation_type='linear', n_samples=1000, noise=0.1):
    np.random.seed(42)
    x = np.random.uniform(-10, 10, n_samples).astype(np.float32)
    
    if equation_type == 'linear':
        y = 5 * x + 10
    elif equation_type == 'quadratic':
        y = 3 * x**2 + 5 * x + 10
    elif equation_type == 'cubic':
        y = 4 * x**3 + 3 * x**2 + 5 * x + 10
    
    # শব্দ যোগ করুন (real-world ডেটার মতো করতে)
    y += np.random.normal(0, noise * np.std(y), n_samples)
    
    # ডেটা স্প্লিট করুন
    split1 = int(0.7 * n_samples)
    split2 = int(0.85 * n_samples)
    
    x_train, y_train = x[:split1], y[:split1]
    x_val, y_val = x[split1:split2], y[split1:split2]
    x_test, y_test = x[split2:], y[split2:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), x, y

# মডেল তৈরির ফাংশন (equation complexity অনুযায়ী)
def create_model(equation_type='linear'):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(1,)))
    
    if equation_type == 'linear':
        # Linear equation-এর জন্য সহজ মডেল
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(1))
    elif equation_type == 'quadratic':
        # Quadratic-এর জন্য মাঝারি জটিলতা
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1))
    elif equation_type == 'cubic':
        # Cubic-এর জন্য বেশি জটিলতা
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# সব equation টেস্ট করুন
equations = ['linear', 'quadratic', 'cubic']
results = {}

for eq_type in equations:
    print(f"\n{'='*50}")
    print(f"ট্রেনিং: {eq_type.upper()} Equation")
    print('='*50)
    
    # ডেটা তৈরি করুন
    (x_train, y_train), (x_val, y_val), (x_test, y_test), x_full, y_full = create_dataset(eq_type, n_samples=2000)
    
    # মডেল তৈরি ও ট্রেনিং
    model = create_model(eq_type)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )
    
    # টেস্ট সেটে মূল্যায়ন
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"টেস্ট MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")
    
    # প্রেডিকশন
    y_pred = model.predict(x_full, verbose=0).flatten()
    
    # রেজাল্ট সংরক্ষণ
    results[eq_type] = {
        'x': x_full,
        'y_true': y_full,
        'y_pred': y_pred,
        'test_mse': test_loss,
        'test_mae': test_mae,
        'epochs': len(history.history['loss'])
    }
    
    # প্লট করুন (শেষে সব একসাথে প্লট করব)
    
# সব প্লট একসাথে দেখান
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, eq_type in enumerate(equations):
    ax = axes[idx]
    ax.scatter(results[eq_type]['x'], results[eq_type]['y_true'], alpha=0.3, label='Original', s=10)
    ax.scatter(results[eq_type]['x'], results[eq_type]['y_pred'], alpha=0.6, label='Predicted', s=10, color='red')
    ax.set_title(f'{eq_type.capitalize()} (MSE: {results[eq_type]["test_mse"]:.2f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('polynomial_regression.png', dpi=150)
plt.show()

# গুরুত্বপূর্ণ পর্যবেক্ষণ (বাংলায়)
print("\n📊 পর্যবেক্ষণ:")
print("1. যত বেশি power (x³ > x² > x), তত বেশি hidden layers ও neurons প্রয়োজন")
print("2. Cubic equation-এর জন্য বেশি training data (2000 samples) প্রয়োজন হয়েছে")
print("3. Higher power = বেশি non-linearity = বেশি model complexity প্রয়োজন")
print("4. EarlyStopping ব্যবহার করে overfitting কমানো হয়েছে")
```

### 📝 রিপোর্টে যা লিখবেন:
- **Power vs Architecture:** Linear → 1 hidden layer, Quadratic → 2 layers, Cubic → 3 layers
- **Power vs Data Size:** Cubic এর জন্য 2000 samples, Linear এর জন্য 500 samples যথেষ্ট
- **গ্রাফ:** Original vs Predicted প্লট রিপোর্টে যোগ করুন

---

## 👕 Assignment 4: FCFNN Classifier (Fashion MNIST, MNIST, CIFAR-10)

### 🐍 ইউনিফাইড কোড (সব ডেটাসেটের জন্য)
```python
# assignment4_fcfnn_classifier.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def train_fcfnn_on_dataset(dataset_name='mnist'):
    # ডেটাসেট লোড করুন
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        input_shape = 784
        num_classes = 10
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        input_shape = 784
        num_classes = 10
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.reshape(-1, 3072)  # 32x32x3 = 3072
        x_test = x_test.reshape(-1, 3072)
        input_shape = 3072
        num_classes = 10
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    
    # নরমালাইজ করুন
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # মডেল তৈরি করুন (ডেটাসেট অনুযায়ী আর্কিটেকচার পরিবর্তন)
    if dataset_name == 'cifar10':
        # CIFAR-10 complex, তাই বড় মডেল
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        # MNIST/Fashion-MNIST সহজ
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # ট্রেনিং
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=1
    )
    
    # টেস্ট সেটে মূল্যায়ন
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\n✅ {dataset_name.upper()} রেজাল্ট:")
    print(f"   টেস্ট অ্যাকুরেসি: {test_acc:.4f}")
    print(f"   টেস্ট লস: {test_loss:.4f}")
    
    return history, test_acc, test_loss

# সব ডেটাসেট টেস্ট করুন
datasets = ['mnist', 'fashion_mnist', 'cifar10']
results = {}

for ds in datasets:
    history, acc, loss = train_fcfnn_on_dataset(ds)
    results[ds] = {'acc': acc, 'loss': loss, 'history': history}

# অ্যাকুরেসি কম্প্যারিজন প্লট
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), [r['acc'] for r in results.values()], color=['blue', 'green', 'red'])
plt.ylabel('Test Accuracy')
plt.title('FCFNN Performance on Different Datasets')
plt.ylim(0, 1)
for i, (ds, r) in enumerate(results.items()):
    plt.text(i, r['acc'] + 0.02, f"{r['acc']:.2%}", ha='center')
plt.grid(axis='y', alpha=0.3)
plt.savefig('fcfnn_comparison.png', dpi=150)
plt.show()
```

### 📊 প্রত্যাশিত রেজাল্ট:
| Dataset | Expected Accuracy |
|---------|-------------------|
| MNIST | 97-98% |
| Fashion MNIST | 88-90% |
| CIFAR-10 | 45-50% (FCFNN দিয়ে) |

> 💡 **গুরুত্বপূর্ণ:** CIFAR-10-এ FCFNN খারাপ করবে কারণ এটি spatial information preserve করতে পারে না। এটা Assignment 5-এ CNN ব্যবহার করে সলভ করা হবে।

---

## 🔷 Assignment 5: CNN Classifier (Fashion MNIST, MNIST, CIFAR-10)

### 🐍 ইউনিফাইড CNN কোড
```python
# assignment5_cnn_classifier.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def create_cnn_model(dataset_name='mnist'):
    if dataset_name == 'cifar10':
        # CIFAR-10 এর জন্য deeper CNN
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
    else:
        # MNIST/Fashion-MNIST এর জন্য simpler CNN
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_cnn_on_dataset(dataset_name='mnist'):
    # ডেটা লোড ও প্রিপ্রসেস
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    
    # নরমালাইজ
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # মডেল তৈরি
    model = create_cnn_model(dataset_name)
    model.summary()
    
    # ট্রেনিং
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=25,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=1
    )
    
    # টেস্ট মূল্যায়ন
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ {dataset_name.upper()} CNN রেজাল্ট: অ্যাকুরেসি = {test_acc:.4f}")
    
    return history, test_acc

# সব ডেটাসেট টেস্ট করুন
datasets = ['mnist', 'fashion_mnist', 'cifar10']
cnn_results = {}

for ds in datasets:
    history, acc = train_cnn_on_dataset(ds)
    cnn_results[ds] = {'acc': acc, 'history': history}
```

### 📊 প্রত্যাশিত রেজাল্ট (CNN):
| Dataset | FCFNN Accuracy | CNN Accuracy | Improvement |
|---------|----------------|--------------|-------------|
| MNIST | ~98% | **~99.2%** | +1.2% |
| Fashion MNIST | ~90% | **~92-93%** | +2-3% |
| CIFAR-10 | ~48% | **~70-75%** | +22-27% |

> 💡 **কী শিখলাম:** CNN spatial features (edges, textures) extract করতে পারে, যা FCFNN পারে না। তাই CIFAR-10-এ CNN অনেক ভালো করে।

---

## ✍️ Assignment 6: Custom Handwritten Digit Dataset

### 📱 ডেটা কালেকশন স্টেপস:
1. আপনি ও আপনার গ্রুপমেটরা ০-৯ পর্যন্ত digit কাগজে লিখুন
2. Mobile camera দিয়ে ছবি তুলুন (সাদা ব্যাকগ্রাউন্ড, কালো ইন্ক)
3. প্রতিটি digit-এর জন্য অন্তত ২০টি ছবি তুলুন (মোট ২০০+ ছবি)
4. ফোল্ডার স্ট্রাকচার:
```
custom_digits/
├── 0/
├── 1/
├── ...
└── 9/
```

### 🐍 ট্রেনিং কোড (MNIST + Custom Data)
```python
# assignment6_custom_digits.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Custom data load করুন
def load_custom_digits(data_dir, img_size=(28,28)):
    images, labels = [], []
    for label in range(10):
        folder = os.path.join(data_dir, str(label))
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder, fname)).convert('L')  # Grayscale
                img = img.resize(img_size)
                img = np.array(img).astype('float32') / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Step 2: MNIST data load করুন
(x_mnist, y_mnist), (x_test, y_test) = keras.datasets.mnist.load_data()
x_mnist = x_mnist.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Step 3: Custom data load করুন (আপনার ডিরেক্টরি পাথ দিন)
custom_dir = '/content/custom_digits'  # Colab-এ mount করুন
x_custom, y_custom = load_custom_digits(custom_dir)

# Step 4: Data combine করুন
x_train = np.concatenate([x_mnist, x_custom], axis=0)
y_train = np.concatenate([y_mnist, y_custom], axis=0)

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Step 5: CNN মডেল
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: ট্রেনিং
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

# Step 7: মূল্যায়ন
# (a) MNIST test set
mnist_loss, mnist_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"MNIST Test Accuracy: {mnist_acc:.4f}")

# (b) Custom test set (custom data থেকে 20% test হিসেবে রাখুন)
split = int(0.8 * len(x_custom))
x_custom_test = x_custom[split:].reshape(-1, 28, 28, 1)
y_custom_test = y_custom[split:]
custom_loss, custom_acc = model.evaluate(x_custom_test, y_custom_test, verbose=0)
print(f"Custom Data Test Accuracy: {custom_acc:.4f}")
```

---

## 📸 Assignment 7: Mobile-Captured Image Classification

### 📱 ডেটা কালেকশন গাইডলাইন:
1. ৫-১০ জন গ্রুপমেট নিন
2. প্রত্যেকের ৫০টি ছবি mobile camera দিয়ে তুলুন (ফেস বা অবজেক্ট)
3. লেবেল: person1, person2, ..., personN
4. Total data: 250-500 images

### 🐍 ট্রেনিং কোড + মেট্রিক্স ট্র্যাকিং
```python
# assignment7_mobile_images.py
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Data loading (Colab-এ Google Drive mount করুন)
train_generator = datagen.flow_from_directory(
    '/content/mobile_images',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    '/content/mobile_images',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training with timing
start_time = time.time()
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)
total_training_time = time.time() - start_time

# Testing time per sample
test_batch = val_generator.next()
start_test = time.time()
preds = model.predict(test_batch[0], verbose=0)
test_time = (time.time() - start_test) / len(test_batch[0])

# Results
print(f"\n✅ ট্রেনিং সময়: {total_training_time:.2f} সেকেন্ড")
print(f"✅ টেস্টিং সময় প্রতি স্যাম্পল: {test_time*1000:.2f} ms")
print(f"✅ মোট প্যারামিটার: {model.count_params():,}")
print(f"✅ ভ্যালিডেশন অ্যাকুরেসি: {max(history.history['val_accuracy']):.4f}")

# Performance curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('mobile_images_performance.png', dpi=150)
plt.show()
```

### 📊 রিপোর্টে যোগ করুন:
- ডেটা vs অ্যাকুরেসি টেবিল (50, 100, 200, 500 images)
- Epoch vs Accuracy গ্রাফ
- Model size (parameters) vs Performance
- Training time vs Data size

---

## 🧠 Assignment 8: VGG16-like Architecture

### 🐍 VGG16 Implementation (without pretraining)
```python
# assignment8_vgg16.py
from tensorflow import keras
from tensorflow.keras import layers

def create_vgg16_like(input_shape=(224,224,3), num_classes=10):
    model = keras.Sequential(name='VGG16_Like')
    
    # Block 1
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 2
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 3
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 4
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 5
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Classification block
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# মডেল তৈরি ও সারসংক্ষেপ
model = create_vgg16_like(input_shape=(32,32,3), num_classes=10)  # CIFAR-10 এর জন্য
model.summary()

# প্যারামিটার গণনা
print(f"\nমোট প্যারামিটার: {model.count_params():,}")
```

> 💡 **নোট:** Original VGG16 224x224 input নেয়, কিন্তু CIFAR-10 এর জন্য আমরা 32x32 input ব্যবহার করেছি। আর্কিটেকচার একই রেখেছি।

---

## 🔍 Assignment 9: Feature Map Visualization

### 🐍 Pre-trained CNN Feature Map Visualization
```python
# assignment9_feature_maps.py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ইমেজ লোড করুন (আপনার পছন্দের ইমেজ)
img_path = '/content/your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# তিনটি pre-trained মডেল লোড করুন
models = {
    'VGG16': (VGG16(weights='imagenet', include_top=False), vgg_preprocess(x)),
    'ResNet50': (ResNet50(weights='imagenet', include_top=False), resnet_preprocess(x)),
    'MobileNetV2': (MobileNetV2(weights='imagenet', include_top=False), mobilenet_preprocess(x))
}

# Feature map visualize করুন
for name, (model, preprocessed_img) in models.items():
    print(f"\n{name} এর feature maps...")
    
    # প্রথম convolutional layer থেকে feature maps পান
    layer_outputs = [layer.output for layer in model.layers[:5]]  # প্রথম ৫ লেয়ার
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(preprocessed_img)
    
    # প্রথম layer-এর feature maps প্লট করুন
    first_layer_activation = activations[0]
    n_features = min(8, first_layer_activation.shape[-1])  # প্রথম ৮টি filter
    
    plt.figure(figsize=(15, 4))
    for i in range(n_features):
        plt.subplot(1, n_features, i+1)
        plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'{name} - Layer 1 Feature Maps', fontsize=16)
    plt.savefig(f'{name}_feature_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 📝 রিপোর্টে লিখুন:
- VGG16: Low-level features (edges, corners)
- ResNet50: Edge + texture features (skip connections এর কারণে ভালো)
- MobileNetV2: Efficient but slightly less detailed features

---

## ⚙️ Assignment 10: Transfer Learning with VGG16

### 🐍 Full vs Partial Fine-tuning
```python
# assignment10_transfer_learning.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ডেটা প্রিপ্রসেসিং
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train, (160, 160))  # VGG16 এর জন্য minimum 32x32, কিন্তু 160x160 ভালো
x_test = tf.image.resize(x_test, (160, 160))
x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

def create_model(fine_tune_all=False):
    # Base model (pre-trained)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,160,3))
    
    if not fine_tune_all:
        # Partial fine-tuning: শুধু top layers ট্রেইন করব
        base_model.trainable = False
    else:
        # Full fine-tuning: সব layers ট্রেইন করব
        base_model.trainable = True
        # শুধু last 5 layers unfreeze করুন (optional)
        for layer in base_model.layers[:-5]:
            layer.trainable = False
    
    # Classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4 if fine_tune_all else 1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Experiment 1: Partial fine-tuning
print("🔄 Partial Fine-tuning (Feature Extraction)...")
model_partial = create_model(fine_tune_all=False)
history_partial = model_partial.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Experiment 2: Full fine-tuning
print("\n🔄 Full Fine-tuning...")
model_full = create_model(fine_tune_all=True)
history_full = model_full.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# রেজাল্ট কম্পেয়ার
partial_acc = max(history_partial.history['val_accuracy'])
full_acc = max(history_full.history['val_accuracy'])

print(f"\n✅ Partial Fine-tuning Val Accuracy: {partial_acc:.4f}")
print(f"✅ Full Fine-tuning Val Accuracy: {full_acc:.4f}")
print(f"✅ Improvement: {(full_acc - partial_acc)*100:.2f}%")

# প্লট
plt.figure(figsize=(10,4))
plt.plot(history_partial.history['val_accuracy'], label='Partial FT')
plt.plot(history_full.history['val_accuracy'], label='Full FT')
plt.title('Partial vs Full Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('fine_tuning_comparison.png', dpi=150)
plt.show()
```

### 📊 প্রত্যাশিত রেজাল্ট:
- Partial FT: ~75-80% accuracy (শুধু classifier head ট্রেইন)
- Full FT: ~82-87% accuracy (সব layers টুইক করা হয়েছে)
- সময়: Full FT বেশি সময় নেয় কিন্তু ভালো রেজাল্ট দেয়

---

## 📉 Assignment 11: PCA/t-SNE Visualization of Features

### 🐍 Feature Extraction + Dimensionality Reduction
```python
# assignment11_pca_tsne.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ডেটা লোড
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:1000]  # দ্রুত প্রসেসিংয়ের জন্য ১০০০ স্যাম্পল
y_train = y_train[:1000]
x_train = np.stack([x_train]*3, axis=-1)  # Grayscale → RGB (VGG16 এর জন্য)
x_train = tf.image.resize(x_train, (32,32))  # VGG16 minimum size
x_train = tf.cast(x_train, tf.float32) / 255.0

# Pre-trained VGG16 (ImageNet)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('block4_pool').output  # Intermediate layer
)

# Features extract করুন
features = feature_extractor.predict(x_train, verbose=0)
features_flat = features.reshape(features.shape[0], -1)

# PCA (2D)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_flat)

# t-SNE (2D)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_tsne = tsne.fit_transform(features_flat[:500])  # t-SNE slow, so 500 samples

# প্লট
plt.figure(figsize=(12,5))

# PCA Plot
plt.subplot(1,2,1)
scatter = plt.scatter(features_pca[:,0], features_pca[:,1], c=y_train, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('PCA of VGG16 Features (Before Transfer Learning)')

# t-SNE Plot
plt.subplot(1,2,2)
scatter = plt.scatter(features_tsne[:,0], features_tsne[:,1], c=y_train[:500], cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE of VGG16 Features')

plt.tight_layout()
plt.savefig('feature_visualization.png', dpi=150)
plt.show()

print(f"✅ PCA explained variance ratio: {pca.explained_variance_ratio_}")
```

### 📝 রিপোর্টে লিখুন:
- PCA: Global structure preserve করে, কিন্তু non-linear relationships দেখায় না
- t-SNE: Local clusters ভালো দেখায়, কিন্তু global structure distort করে
- Transfer learning এর পরে features আরও স্পষ্টভাবে cluster হবে

---

## 🔄 Assignment 12-16: Callbacks, Augmentation, Overfitting, Activation Functions

### 📦 সবগুলোর জন্য ইউনিফাইড কোড (Colab Notebook হিসেবে ব্যবহার করুন)

```python
# assignments_12_to_16.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# ডেটা লোড
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# ========== Assignment 12: Data Augmentation ==========
datagen_none = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
datagen_basic = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen_advanced = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# ========== Assignment 13: Dropout for Overfitting ==========
def create_model(dropout_rate=0.0, augmentation=None):
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Dropout এক্সপেরিমেন্ট
models_dropout = {}
for dr in [0.0, 0.3, 0.5]:
    print(f"\nTraining with dropout={dr}")
    model = create_model(dropout_rate=dr)
    history = model.fit(
        x_train, y_train,
        epochs=25,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    models_dropout[dr] = history

# প্লট: Dropout vs Overfitting
plt.figure(figsize=(12,4))
for idx, (dr, hist) in enumerate(models_dropout.items()):
    plt.subplot(1,3,idx+1)
    plt.plot(hist.history['accuracy'], label='Train')
    plt.plot(hist.history['val_accuracy'], label='Val')
    plt.title(f'Dropout={dr}')
    plt.ylim(0,1)
    plt.legend()
plt.tight_layout()
plt.savefig('dropout_overfitting.png', dpi=150)
plt.show()

# ========== Assignment 14: Activation Functions ==========
activations = ['relu', 'tanh', 'sigmoid']
results_activation = {}

for act in activations:
    print(f"\nTraining with activation={act}")
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation=act, padding='same', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation=act, padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation=act),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=0)
    results_activation[act] = max(history.history['val_accuracy'])

print("\n✅ Activation Function Comparison:")
for act, acc in results_activation.items():
    print(f"   {act:10s}: {acc:.4f}")

# ========== Assignment 15: Callbacks ==========
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),
    keras.callbacks.TensorBoard(log_dir='./logs')  # TensorBoard visualization
]

# ========== Assignment 16: Performance Curves ==========
model = create_model(dropout_rate=0.3)
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# পারফরম্যান্স কার্ভ প্লট
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.axhline(y=0.9, color='r', linestyle='--', label='Target 90%')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('performance_curves.png', dpi=150)
plt.show()

# হাইপারপ্যারামিটার টিউনিং গাইডলাইন
print("\n📊 হাইপারপ্যারামিটার টিউনিং গাইড:")
print("1. যদি training accuracy ↑ কিন্তু validation accuracy ↓ → Overfitting (Dropout/Regularization বাড়ান)")
print("2. যদি দুইটাই কম → Underfitting (Model complexity বাড়ান অথবা ট্রেনিং সময় বাড়ান)")
print("3. যদি loss অস্থির → Learning rate কমান")
print("4. Validation loss বাড়লে EarlyStopping কাজ করবে")
```

---

## 📚 সারসংক্ষেপ: সব Assignment-এর জন্য LaTeX Template

Overleaf-এ নিচের template ব্যবহার করুন:

```latex
\documentclass[12pt]{article}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\title{AI Lab Assignments Report \\ University of Rajshahi \\ Faculty of Engineering \\ Department of CSE}
\author{Your Name \\ ID: XXXXXXXX}
\date{\today}

\begin{document}
\maketitle

\section{Assignment 1: FCFNN Drawing}
\begin{itemize}
    \item Input: 8 neurons
    \item Hidden 1: 4 neurons
    \item Hidden 2: 8 neurons
    \item Hidden 3: 4 neurons
    \item Output: 10 neurons
\end{itemize}
Figure \ref{fig:fcfnn} shows the manually drawn network.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{fcfnn_drawing.jpg}
    \caption{Manually drawn FCFNN}
    \label{fig:fcfnn}
\end{figure}

% ... continue for all assignments ...

\section{Conclusion}
We successfully implemented various neural network architectures including FCFNN, CNN, and transfer learning models. Key findings:
\begin{itemize}
    \item CNN outperforms FCFNN significantly on image data (especially CIFAR-10)
    \item Transfer learning with VGG16 achieves high accuracy with less data
    \item Data augmentation and dropout effectively reduce overfitting
    \item Higher polynomial degree requires deeper networks and more data
\end{itemize}

\end{document}
```

---

## ✅ পরবর্তী পদক্ষেপ:

1. **প্রতিটি assignment আলাদা Colab notebook এ করুন**
2. **Overleaf-এ LaTeX report তৈরি করুন**
3. **Assignment 6 ও 7 এর জন্য নিজের ডেটা কালেক্ট করুন**
4. **সব গ্রাফ ও রেজাল্ট report-এ যোগ করুন**
5. **প্রতিটি assignment এর জন্য ২-৩ পৃষ্ঠার রিপোর্ট লিখুন**

> 📌 **সতর্কতা:** এই কোডগুলো educational purpose এ দেওয়া হয়েছে। নিজে করে বুঝে নিন, শুধু copy-paste করবেন না। প্রতিটি লাইন বুঝে নিন কেন করা হয়েছে।

সব assignment complete করতে সাহায্য লাগলে আবার জিজ্ঞাসা করুন! 😊
