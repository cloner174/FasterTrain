# FasterTrain
Faster R-CNN Trainer: A project for experimenting with the Faster R-CNN model. This is A library for training Faster R-CNN models with a MobileNetV3 backbone on custom datasets


## Features

- **Model Loading and Customization**: Utilizes `fasterrcnn_mobilenet_v3_large_320_fpn`.
- **Data Preprocessing**: Supports COCO format with customizable augmentations.
- **Configurable Training Pipeline**: Managed via `arg.yaml`.
- **Metrics Logging**: Tracks training and validation losses, precision, recall, and mAP.
- **Checkpointing**: Saves the best and latest model weights.
- **Visualization**: Generates plots of training metrics.
- **Mixed Precision Training**: Supports AMP for faster training.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/cloner174/FasterTrain.git
    cd FasterTrain
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Edit `data.yaml`**: Specify your dataset paths and class names.

    ```yaml
    train: ./data/train
    val: ./data/val
    nc: 20
    names: ['class1', 'class2', ..., 'class20']
    ```

2. **Edit `arg.yaml`**: Configure training parameters as needed.

    ```yaml
    task: detect
    mode: train
    model: fasterrcnn_mobilenet_v3_large_320_fpn
    pretrained: true
    data: data.yaml
    epochs: 100
    batch: 8
    imgsz: 320
    device: 0
    workers: 4
    optimizer: SGD
    lr0: 0.01
    momentum: 0.937
    weight_decay: 0.0005
    amp: true
    patience: 20
    save_dir: faster_rcnn_project/train
    # ... other parameters
    ```

## Usage

1. **Prepare Dataset**: Organize your dataset in COCO format.

    ```
    data/
    ├── train/
    │   ├── images/
    │   └── annotations.json
    └── val/
        ├── images/
        └── annotations.json
    ```

2. **Run Training**:

    ```bash
    python trainer.py --config arg.yaml
    ```

3. **Monitor Training**:

    - **Checkpoints**: Located in `save_dir` (e.g., `faster_rcnn_project/train/best_weights.pth`).
    - **Metrics**: Saved as `metrics.json`.
    - **Plots**: Generated as `metrics.png`.

## License

MIT License
