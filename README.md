# Multi-Digit House Number Recognition with SVHN

A deep learning project for recognizing house numbers containing one to four digits from real Street View House Numbers (SVHN) images.

The project uses the original SVHN images and bounding-box annotations to isolate complete house numbers before training an EfficientNetV2-based model. The model predicts both the number of digits and each digit position.

## Results

The final model achieved:

- **Exact sequence accuracy:** 75.40%
- **Character accuracy:** 84.65%
- **Number-length accuracy:** 95.51%

### Accuracy by Number Length

| Number Length | Accuracy |
|---|---:|
| 1 digit | 87.07% |
| 2 digits | 75.75% |
| 3 digits | 62.47% |
| 4 digits | 41.10% |

## Approach

The workflow includes:

- loading the original SVHN street-number images;
- reading the official `digitStruct` bounding-box annotations;
- cropping the complete visible house number;
- preserving image aspect ratio during resizing;
- training an EfficientNetV2B0 feature extractor;
- predicting sequence length separately from digit identity;
- using position-specific digit classifiers;
- evaluating performance at both sequence and digit level.

The training process also accounts for differences in the frequency of one-, two-, three-, and four-digit examples.

## Evaluation

The notebook reports:

- exact sequence accuracy;
- character accuracy;
- number-length accuracy;
- macro sequence accuracy;
- accuracy by number length;
- digit accuracy by position;
- digit-level confusion matrix;
- number-length confusion matrix;
- sample test predictions;
- training and validation curves;
- total runtime.

## Dataset

This project uses the **Street View House Numbers (SVHN)** dataset.

The notebook checks whether the dataset is already available in the environment and downloads it automatically when required.

## Model

The recognition model is built with TensorFlow/Keras and uses:

- EfficientNetV2B0
- transfer learning with ImageNet weights
- data augmentation
- shared image features
- a sequence-length prediction head
- four position-specific digit prediction heads
- staged fine-tuning

## Running the Notebook

Open:

`svhn-multidigit-recognition.ipynb`

The notebook is designed to run from start to finish without requiring manual dataset preparation.

A GPU is recommended because training EfficientNetV2 on the full dataset can take a significant amount of time and compute resources.

## Conclusion

The model achieved **75.40% exact sequence accuracy**, **84.65% character accuracy**, and **95.51% number-length accuracy** on the SVHN test set.

Accuracy was **87.07% for one digit**, **75.75% for two digits**, **62.47% for three digits**, and **41.10% for four digits**.

The results show that recognition becomes more difficult as sequence length increases, while the model maintains strong digit-level and number-length prediction performance.
