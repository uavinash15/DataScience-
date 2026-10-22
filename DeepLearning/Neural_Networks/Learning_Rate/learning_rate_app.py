import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

st.set_page_config(page_title="Learning Rate Study", layout="wide")
st.title("Effect of learning rate on accuracy (blobs problem)")
st.write(
    "A small neural network (50 ReLU units, softmax output) is trained with SGD "
    "at different learning rates. Compare the train and test accuracy curves."
)


@st.cache_data
def prepare_data():
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2,
                      cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train = 500
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def fit_model(trainX, trainy, testX, testy, lrate, epochs):
    keras.utils.set_random_seed(1)
    model = Sequential([
        Input(shape=(2,)),
        Dense(50, activation="relu", kernel_initializer="he_uniform"),
        Dense(3, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=SGD(learning_rate=lrate),
                  metrics=["accuracy"])
    history = model.fit(trainX, trainy, validation_data=(testX, testy),
                        epochs=epochs, verbose=0)
    return history.history


# ---------- sidebar controls ----------
st.sidebar.header("Settings")
all_rates = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
chosen = st.sidebar.multiselect(
    "Learning rates", all_rates, default=[1e-0, 1e-2, 1e-4, 1e-6],
    format_func=lambda x: f"{x:.0e}",
)
epochs = st.sidebar.slider("Epochs", 20, 200, 100, step=20)
run = st.sidebar.button("Train models", type="primary")

# ---------- main ----------
if run:
    if not chosen:
        st.warning("Select at least one learning rate.")
    else:
        trainX, trainy, testX, testy = prepare_data()
        chosen = sorted(chosen, reverse=True)
        cols = 2
        rows = (len(chosen) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), squeeze=False)

        progress = st.progress(0.0, text="Training...")
        for i, lr in enumerate(chosen):
            hist = fit_model(trainX, trainy, testX, testy, lr, epochs)
            ax = axes[i // cols][i % cols]
            ax.plot(hist["accuracy"], label="train")
            ax.plot(hist["val_accuracy"], label="test")
            ax.set_title(f"lrate={lr}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("accuracy")
            ax.legend()
            progress.progress((i + 1) / len(chosen), text=f"Finished lr={lr}")

        # hide unused subplot slots
        for j in range(len(chosen), rows * cols):
            axes[j // cols][j % cols].axis("off")

        fig.tight_layout()
        progress.empty()
        st.pyplot(fig)
else:
    st.info("Pick learning rates and epochs in the sidebar, then click **Train models**.")
