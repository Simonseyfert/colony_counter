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
