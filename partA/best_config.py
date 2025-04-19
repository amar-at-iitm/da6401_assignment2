# Configuration matching best model (update these as needed)
best_config = {
    "filters_per_layer": [32, 64, 64, 128, 128],
    "activation": "GELU",
    "dropout_rate": 0.3,
    "use_batchnorm": True,
    "input_shape": (3, 256, 256),
    "batch_size": 32,
    "model_path": "best_model.pth"    
}
