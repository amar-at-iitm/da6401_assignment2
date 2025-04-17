# Part B : Fine-tuning a pre-trained model
This section of the assignment focuses on **fine-tuning GoogLeNet**, a deep CNN pre-trained on the **ImageNet** dataset, for a custom 10-class classification task using the **iNaturalist_12K** dataset.

## Objectives
- Load a **pre-trained GoogLeNet** model from `torchvision.models`.
- Adapt it to our dataset by resizing inputs and modifying the output layer.
- Apply different **fine-tuning strategies** (`only_fc`, `partial`, `all`).
- Use **wandb sweeps** to explore which strategy performs best.
- Implement **early stopping** to prevent overfitting.
- Save the best model based on **validation accuracy** across all sweep runs.
- Evaluate the fine-tuned model on the test set.

## File Structure

```
partB/
│
├── finetune_model.py          # Fine-tuning script with wandb sweep
├── sweep_config_2.py          # wandb sweep configuration for Part B
├── best_model_B.pth           # Saved best model from fine-tuning (after sweep)
├── best_accuracy_B.txt        # Stores the highest validation accuracy across runs
└── README.md                  # This file
```
####  Setup
 Change the direcotry:
   ```bash
   cd partB
   ```
#### Run the Script

```bash
python finetune_model.py
```

This will:
- Start a wandb sweep with `sweep_config_2`
- Automatically run 5–10 experiments
- Log metrics and predictions to wandb.ai

## Fine-tuning Strategies Compared

| Strategy     | Description                                           |
|--------------|-------------------------------------------------------|
| `only_fc`    | Freeze all layers except the final classifier         |
| `partial`    | Unfreeze only the last few layer blocks               |
| `all`        | Fine-tune the **entire** GoogLeNet network            |

In our wandb sweep:
- **`all`** gave the **best validation accuracy** in 8 out of 10 runs 

## Key Takeaways
- Fine-tuning **all layers** of GoogLeNet gave the best results
- Pre-trained ImageNet features transfer well to iNaturalist
- wandb sweeps helped in hyperparameter selection efficiently


