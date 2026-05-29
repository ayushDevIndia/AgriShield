import os
import json

MODELS_DIR = "/home/pirates/Desktop/Project Crop/models"

notebooks = [
    ("LNet.ipynb", "LNet.h5", "LNet"),
    ("DenseNet.ipynb", "DenseNet.h5", "DenseNet"),
    ("InceptionV3.ipynb", "InceptionV3.h5", "InceptionV3"),
    ("Inception_ResNet_v2.ipynb", "Inception_ResNet_v2.h5", "Inception-ResNet-v2"),
    ("Untitled65.ipynb", "Untitled65.h5", "ConvNeXt-Tiny")
]

print("Starting to inject Keras save weights cell into Jupyter Notebooks...")

for filename, save_name, model_display in notebooks:
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        continue
        
    try:
        # Load JSON notebook
        with open(filepath, "r", encoding="utf-8") as f:
            notebook_data = json.load(f)
            
        # Verify it has cells
        if "cells" not in notebook_data:
            print(f"ERROR: No cells found in {filename}")
            continue
            
        # Check if we already added a save cell to prevent duplication
        already_has_save = False
        for cell in notebook_data["cells"]:
            if cell.get("cell_type") == "code":
                source_lines = "".join(cell.get("source", []))
                if f'model.save("{save_name}")' in source_lines or f"model.save('{save_name}')" in source_lines:
                    already_has_save = True
                    break
                    
        if already_has_save:
            print(f"Skipping {filename}: Model save cell already exists in the notebook.")
            continue
            
        # Create new code cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "collapsed": False,
                "scrolled": False
            },
            "outputs": [],
            "source": [
                "# ============================================================\n",
                "# SAVE TRAINED DEEP LEARNING MODEL TO DISK\n",
                "# ============================================================\n",
                f"# This cell exports the fully-trained model and weights as an .h5 file\n",
                f"model.save(\"{save_name}\")\n",
                f"print(\"Trained {model_display} model successfully saved as {save_name}!\")"
            ]
        }
        
        # Append cell to notebook
        notebook_data["cells"].append(new_cell)
        
        # Write back to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=2, ensure_ascii=False)
            
        print(f" -> SUCCESS: Appended model.save cell to {filename} for exporting {save_name}.")
        
    except Exception as e:
        print(f"ERROR: Failed to update {filename} due to: {e}")

print("\nNotebook injection completed!")
