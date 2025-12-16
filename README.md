# colony_counter

Automatic classification of virus-infected cell cultures from well-plate images.

This repository contains the executable files (for **macOS**, **Linux**, and **Windows**) to run the automatic bacteria / cell-death classifier without requiring any interaction with Python.

---

## Authors

- **Simon Seyfert**  
- **Lina Groß**  
- **Vidhi Oswal**

For questions or issues, feel free to contact:  
📧 **simon.seyfert@epfl.ch**

---

## How to install
0. Note that this installation guide is only tested on/for windows. 
1. Click on the latest release
![Overview Github](data/images_readme/image-1.png)

2. Click on the "best_model_robust_aug.joblib" to start the download. 
For windows: Click on ColonyCounter-windows.exe to start the download.
![Download](data/images_readme/image.png)

3. Place the two files in the same folder. If not, the model will not be found.
![Folder structure](data/images_readme/folder_structure.png)

## How to run
1. When running for the first time the following appears. First click on "More info", then select "Run Anyway"
![run-windows](data/images_readme/run_windows.png)

2. The following window appears. It can be resized like any other windows program
![colony_counter](data/images_readme/colony_counter.png)

## Repository Structure (Preliminary)

```text
colony_counter/
│
├── executables/            # Standalone executables (no Python required)
│   ├── windows/
│   ├── macos/
│   └── linux/
│
├── src/                    # Python source code
│   ├── inference/          # Model inference logic
│   ├── preprocessing/     # Image loading & preprocessing
│   ├── models/             # Trained models / checkpoints
│   └── gui/                # GUI application code
│
├── data/                   # Example input images (optional)
│
├── notebooks/              # Development & analysis notebooks
│
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment (optional)
└── README.md
