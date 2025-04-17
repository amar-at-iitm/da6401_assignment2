# DA6401_assignment2
---
#### `Amar Kumar`  `(MA24M002)`
#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`
##### For more detail go to [wandb project report](https://wandb.ai/amar74384-iit-madras/DA6401_assign_2/reports/DA6401-Assignment-2--VmlldzoxMjA0Njg2Ng?accessToken=qkpn51rke34k3nyepwmf0aukpkcrdwq8tattbiaq61jyfvjis6dq0b5jiddgiowb)
---
### Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/amar-at-iitm/da6401_assignment2
   cd da6401_assignment2
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure WandB:
   ```bash
   wandb login
   ```

### Project Structure 
```
.
├── requirements.txt              # List of Python dependencies
├── README.md                     # Root README with project overview
├── datapreparation.py            # Downloads and prepares the data

├── partA/                        # Part A: CNN trained from scratch
│   ├── model.py                  # Custom CNN model with 5 conv layers
│   ├── train.py                  # Training with wandb sweeps
│   ├── test_model.py             # Evaluate best model and show predictions
│   ├── sweep_config.py           # Hyperparameter sweep config for wandb
│   ├── best_config.py            # Best configuration from sweep
│   ├── best_model.pth            # Saved best model weights
│   ├── best_accuracy.txt         # Validation accuracy of best model
│   └── README.md                 # README for Part A

├── partB/                        # Part B: Fine-tuning pre-trained GoogLeNet
│   ├── fine_tune_model.py         # Fine-tuning with wandb sweep 
│   ├── sweep_config_2.py         # Sweep config for Part B
│   ├── best_model_B.pth          # Best model after sweep
│   ├── best_accuracy_B.txt       # Validation accuracy of best model
│   └── README.md                 # README for Part B (fine-tuning overview)

└── inaturalist_12K/              # Dataset folder (train, val, test)
    ├── train/
    ├── val/
    └── test/

```
---
## [Part A: Training from scratch](https://github.com/amar-at-iitm/da6401_assignment2/tree/main/partA) 

CNN based image classifiers using a subset of the iNaturalist dataset.

- Build a CNN with:
   - 5 Conv → Activation → MaxPool blocks
   - Customizable dense & output layers (10 classes)
   - Flexible filters, kernel sizes, activations, and neurons
- Compute total parameters & operations (based on m, k×k, n)
- Train on iNaturalist:
   - Use 80/20 train-validation split (balanced by class)
   - Apply WandB sweeps for hyperparameter tuning:
      - Filters, activations, dropout, batch norm, etc.
- Include:
   - Accuracy vs experiments plot
   - Parallel coordinates & correlation summary
- Report test accuracy
- Show results in a creative 10×3 prediction grid.
---
## [Part B : Fine-tuning a pre-trained model](https://github.com/amar-at-iitm/da6401_assignment2/tree/main/partB)
Unlike Part A, where I trained a CNN from scratch, this section explores how leveraging pre-trained models can improve performance and reduce training time.

Key highlights:
- Loaded and adapted GoogLeNet using torchvision.models
- Resized dataset images to 224×224 to match ImageNet input requirements
- Replaced the final classifier layer to output 10 classes
- Implemented three fine-tuning strategies:
   - only_fc: train only the final fully connected layer
   - partial: unfreeze the last few layers
   - all: train the entire network
- Performed a wandb sweep to find the best strategy and hyperparameters
- Saved the best model across all sweep runs based on validation accuracy
- Evaluated final model on unseen test data, with predictions logged to wandb

Best results were achieved by fine-tuning the entire network (freeze_strategy = "all"), confirming the power of transfer learning with high-quality pretrained features.

--- 
## Author
**Name:** *Amar Kumar*  
**Course:** *DA6401 - Deep Learning*  
**Institute:** *IIT Madras*

