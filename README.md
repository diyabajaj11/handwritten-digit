## Handwritten Digit Recognition Pipeline

This project demonstrates how to build a simple handwritten digit recognition system from scratch using OpenCV and Scikit-learn.
We take a scanned sheet of handwritten digits, automatically extract each digit, and train a machine learning model to classify them.

# Features
Convert scanned handwritten sheet into black & white.
Segment each digit using contour detection.
Resize digits into 28×28 images (MNIST-style).
Save digits into a dataset folder.
Train a classifier (Logistic Regression) on the custom dataset.
Evaluate accuracy on unseen digits.

# Extract digits from a sheet
python extract_digits.py
This will process one_two_three.jpeg and save each digit into digits_dataset/.

# Train the model
python train_model.py
This will:
Load digit images from digits_dataset/
Preprocess them (resize,flatten,normalize)
Train a Logistic Regression model
Print accuracy on a test set

Using raw pixel features : ~50–60% accuracy

# Future Improvements
Use CNN (Convolutional Neural Networks) for higher accuracy.
Add support for more digit classes (0–9).
