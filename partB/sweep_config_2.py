sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "freeze_strategy": {
            "values": ["only_fc", "partial", "all"]
        },
        "batch_size": {
            "values": [32, 64]
        },
        "learning_rate": {
            "values": [1e-4, 5e-5, 1e-5]
        },
        "epochs": {
            "values": [10, 15]
        },
        "img_size": {
            "values": [224]
        }
    }
}
