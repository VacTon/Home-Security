# 🎓 How to Train Your Own Face Recognition Model on Kaggle

This guide explains how to use the `kaggle_arcface_script.py` to train a custom model for your Raspberry Pi.

## Step 1: Prepare Your Data

1.  **Collect Photos:** Organized in folders by person (e.g., `dataset/Person1`, `dataset/Person2`).
2.  **Add a Public Dataset:** Since Deep Learning needs thousands of faces to learn *generic* facial features, you should mix your small dataset with a public one like **WiderFace** or **LFW** (Labeled Faces in the Wild).
3.  **Zip It:** Compress your `dataset` folder into `dataset.zip`.

## Step 2: Set Up Kaggle

1.  Log in to [Kaggle.com](https://www.kaggle.com).
2.  Click **Create** -> **New Notebook**.
3.  **IMPORTANT:** In the right sidebar, verify these settings:
    *   **Accelerator:** GPU T4 x2 (or P100)
    *   **Internet:** On (Switch this on in Settings)

## Step 3: Upload Data

1.  In the Notebook sidebar, click **Add Data**.
2.  Click **Upload** and select your `dataset.zip`.
3.  Wait for the upload to finish. It will appear under `/kaggle/input`.

## Step 4: Run the Code

1.  Open the file `training/kaggle_arcface_script.py` in this repository.
2.  **Copy the entire content.**
3.  Paste it into the first cell of your Kaggle Notebook.
4.  **Find & Update specific line:**
    ```python
    "dataset_path": "/kaggle/input/your-dataset-name"  # <--- CHANGE THIS PART
    ```
    (You can copy the exact path from the Kaggle sidebar by clicking the Copy icon next to your dataset folder).

## Step 5: Train & Download

1.  Click the **Run** (Play) button.
2.  Watch the progress bar! It will train for 20 epochs.
3.  When finished, the script will create `custom_arcface.onnx`.
4.  Download this file and copy it to your Raspberry Pi project folder (`models/`).

## Why this works?

This script implements **ArcFace** (Additive Angular Margin Loss), which is state-of-the-art. It forces the model to not just "classify" faces (which doesn't work for new people) but to learn a **512-dimension vector space** where same-person faces are clustered tightly together.
