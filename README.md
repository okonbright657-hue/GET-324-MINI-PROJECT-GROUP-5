# 🍎 Fresh vs Rotten Apple Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies apple images as **Fresh** or **Rotten**, with a Streamlit web app for interactive inference.

> **Data integrity notice:** The source dataset contains pre-augmented images (`rotated_by_X_`, `translation_`, `vertical_flip_`, `saltandpepper_` prefixes) split across `train/`, `validation/`, and `test/`. **

## Project Structure

```
.
├── api
     └── app.py             # Streamlit inference app
├── requirements.txt        # Python dependencies
├── models/
│   └── custom_cnn.keras    # Trained model (add after training — not committed by default)
└── README.md
└── results                 # outcome of trained model(confusion matrix, learning curve)
```

## Dataset

- **Source:** [Fresh Apple vs Rotten Apple Classification](https://www.kaggle.com/datasets/srishtisharma9977/fresh-apple-vs-rotten-apple-classification) (Kaggle)
- **Classes:** `Fresh`, `Rotten`
- **Structure:** `train/`, `validation/`, `test/`, each with `Fresh/` and `Rotten/` subfolders
- **Note:** Images include augmented variants (rotation, translation, flip, salt-and-pepper noise) of an unknown, smaller number of original photos — see the data integrity notice above.


## Model

Two architectures were compared:

1. **Custom CNN** — trained from scratch. 3 convolutional blocks (Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout), `GlobalAveragePooling2D`, dense head, single sigmoid output.
2. **Transfer learning (MobileNetV3Small)** — _(if/when completed)_ pretrained on ImageNet, fine-tuned on this dataset.

| Setting | Value |
|---|---|
| Input size | 128 × 128 × 3 |
| Batch size | 32 |
| Loss | Binary cross-entropy |
| Output | Single sigmoid neuron (0 = Fresh, 1 = Rotten) |
| Optimizer | Adam |

## Results

_Fill in only after the data integrity check above is confirmed clean._

| Model | Val Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Custom CNN | 0.9921 | 0.9989 | 0.9981 | 1.0000 | 0.9990 |

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

Place your trained model at `models/custom_cnn.keras` (not included in this repo by default — see note in `.gitignore` considerations below).

## Running the App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload an apple image, and view the prediction with confidence score.

## Deployment

1. Push `app.py`, `requirements.txt`, and `models/custom_cnn.keras` to a GitHub repo.
   - Check model file size first: `ls -lh models/custom_cnn.keras`. If it's over ~100 MB, GitHub will reject a plain push — use [Git LFS](https://git-lfs.com/) instead.
2. Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) (or any host that runs `streamlit run app.py`), pointing at `app.py`.

## License

    ` MIT License

    Copyright (c) 2026 [GET324 - FE5]

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE. ``