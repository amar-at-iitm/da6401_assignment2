# DA6401_assignment2
#### `Amar Kumar`  `(MA24M002)`
#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`
##### For more detail go to [wandb project report](https://wandb.ai/amar74384-iit-madras/DA6401_assign_2/reports/DA6401-Assignment-2--VmlldzoxMjA0Njg2Ng?accessToken=qkpn51rke34k3nyepwmf0aukpkcrdwq8tattbiaq61jyfvjis6dq0b5jiddgiowb)

# CNN based image classifiers using a subset of the iNaturalist dataset.

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
### `data_preparation.py`
- Downloads the Nature 12K dataset
- Unzips the downloaded file
- Renames the original val/ folder to test/
- Creates a new `val` folder containing 20% of images randomly moved from the `train` folder.
- Resizes all the images in each folder to 256*256
- Deletes unnecessary files( if exist) to avoid errors while training, validation and testing

#### Dataset Structure After Processing
```
.
inaturalist_12K/
├── train/    # Currently, 80% of original 
├── val/      # 20% split from train
└── test/     # originally 'val/'
```
