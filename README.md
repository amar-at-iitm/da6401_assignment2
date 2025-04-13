# DA6401_assignment2
#### `Amar Kumar`  `(MA24M002)`
#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`
##### For more detail go to [wandb project report](https://wandb.ai/amar74384-iit-madras/DA6401_assign_2/reports/DA6401-Assignment-2--VmlldzoxMjA0Njg2Ng?accessToken=qkpn51rke34k3nyepwmf0aukpkcrdwq8tattbiaq61jyfvjis6dq0b5jiddgiowb)

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
```
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

## [Part B : Fine-tuning a pre-trained model](https://github.com/amar-at-iitm/da6401_assignment2/tree/main/partB)
- Loads a pre-trained model (e.g., ResNet50, VGG, EfficientNetV2, ViT) from torchvision, trained on ImageNet.
- Fine-tunes it on iNaturalist dataset instead of training from scratch.
