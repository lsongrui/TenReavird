# TenReavird Training Pipeline on Google Colab

This document outlines the step-by-step process for setting up the environment and running the TenReavird training pipeline in a Google Colab notebook.

1.  **Open the File Browser:** In your Colab notebook, click the folder icon on the left sidebar.
2.  **Upload Files:**
      * Make sure you are in the `/content/` directory.
      * Drag and drop `PressurePointCloud.zip` and `Pressure_pseudo_VTK.zip` from your computer into the file browser. Wait for the uploads to complete.
3.  **Run Commands in Colab Cells:**
      * **Cell 1: Unzip Files**
        ```bash
        unzip -q /content/PressurePointCloud.zip -d /content/
        unzip -q /content/Pressure_pseudo_VTK.zip -d /content/
        ```
      * **Cell 2: Clone Repo**
        ```bash
        git clone https://github.com/lsongrui/TenReavird.git
        ```
      * **Cell 3: Install Dependencies**
        ```python
        %cd TenReavird/RegDGCNN_SurfaceFields/
        !pip install -q -r requirements.txt
        ```
      * **Cell 4: Run Training**
        ```bash
        python run_pipeline.py --stages train --exp_name "DrivAerNet_Pressure" --dataset_path "/content/Pressure_pseudo_VTK" --subset_dir "../train_val_test_splits/" --cache_dir "/content/PressurePointCloud" --num_points 10000 --batch_size 12 --epochs 150 --gpus "0" --num_best_models 10
        ```