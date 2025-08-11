# MDCC

Official implementation of our paper:

**"Feature mixing-driven dynamic contrastive consistency learning for robust source-free unsupervised domain adaptation"**

*(Submitted to The Visual Computer)*

## Framework

![MDCC流程图](/figs/pipeline.png)

## Installation

 We use Python 3.10.14 environment with PyTorch 2.2.2+CUDA 11.8.

```python
pip install -r requirements.txt
```

## Prepare datasets

Please download [Office31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA)，[Office-Home](https://github.com/jindongwang/transferlearning/tree/master/data#office-home)，[VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)，[DomainNet-126 (cleaned version)](https://ai.bu.edu/M3SDA/).

We have organized the files as follows:

<details>
<summary>📂 Dataset & Project Structure (Click to Expand)​</summary>


```text
<root>/
├── DATASETS/
│   ├── office/                 
│   │   ├── amazon/
│   │   │   ├── back_pack/
│   │   │   │   ├── frame_0001.jpg
│   │   │   │   ├── frame_0002.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── dslr/
│   │   ├── webcam/
│   │   ├── amazon_list.txt
│   │   ├── dslr_list.txt
│   │   └── webcam_list.txt
│   │
│   ├── office-home/
│   │   ├── Art/
│   │   ├── Clipart/
│   │   ├── Product/
│   │   ├── RealWorld/
│   │   ├── Art_list.txt
│   │   ├── Clipart_list.txt
│   │   ├── Product_list.txt
│   │   └── RealWorld_list.txt
│   │
│   ├── VISDA-C/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── train_list.txt
│   │   └── validation_list.txt
│   │
│   └── domainnet126/
│       ├── clipart/
│       ├── painting/
│       ├── real/
│       ├── sketch/
│       ├── clipart_list.txt
│       ├── painting_list.txt
│       ├── real_list.txt
│       └── sketch_list.txt
│
└── MDCC/
    └── ...
```

</details> ```

You can use the `generate_list.py` file to generate your own `.txt` files. Note that for `DomainNet-126`, the `.txt` files we use are from [AdaContrast](https://github.com/DianCh/AdaContrast/tree/master/datasets/domainnet-126). If no directory adjustment is needed, feel free to use our generated `.txt` files available at [[here](https://drive.google.com/drive/folders/1-RUqQqfLEcO8d8YaIJY0pL6A8QhMMGSQ?usp=drive_link)].

## Train source model

You have two options:
1. Use our pre-trained models, they are available at [[Google Drive](https://drive.google.com/drive/folders/1ahBN5-MKOihYa69Ae9OA-IeLJnUiUqVD?usp=drive_link)]
2. Train your own models

To train your own models, please run the following commands.

### Office31

```python
python train_source.py --dset office --net resnet50 --max_epoch 100 --batch_size 64 --lr 1e-2 --s 0
python train_source.py --dset office --net resnet50 --max_epoch 100 --batch_size 64 --lr 1e-2 --s 1
python train_source.py --dset office --net resnet50 --max_epoch 100 --batch_size 64 --lr 1e-2 --s 2
```

### Office-Home

```python
python train_source.py --dset office-home --net resnet50 --max_epoch 50 --batch_size 64 --lr 1e-2 --s 0
python train_source.py --dset office-home --net resnet50 --max_epoch 50 --batch_size 64 --lr 1e-2 --s 1
python train_source.py --dset office-home --net resnet50 --max_epoch 50 --batch_size 64 --lr 1e-2 --s 2
python train_source.py --dset office-home --net resnet50 --max_epoch 50 --batch_size 64 --lr 1e-2 --s 3
```

### VisDA-C

```python
python train_source.py --dset VISDA-C --net resnet101 --max_epoch 10 --batch_size 64 --lr 1e-3 --s 0
```

### DomainNet-126

```python
python train_source.py --dset domainnet126 --net resnet50 --max_epoch 30 --batch_size 64 --lr 1e-2 --s 0
python train_source.py --dset domainnet126 --net resnet50 --max_epoch 30 --batch_size 64 --lr 1e-2 --s 1
python train_source.py --dset domainnet126 --net resnet50 --max_epoch 30 --batch_size 64 --lr 1e-2 --s 2
python train_source.py --dset domainnet126 --net resnet50 --max_epoch 30 --batch_size 64 --lr 1e-2 --s 3
```

After training, you'll get `source_F.pt`, `source_B.pt`, and `source_C.pt`, saved by default in `san/uda/{dset}/{names[s]}`. Manually move/copy them to `{output_src}` (default: `res/ckps/source`) or change `{output_src}`.

## Train target model

To keep it concise, we only list the commands for the first task in each dataset. You can easily change the source (`--s`) and target (`--t`) domains as needed.

### Office31 (amazon, dslr, webcam)

```python
# Please change `--s` and `--t` to run other tasks
# amazon -> dslr
python train_target.py --dset office --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 0 --t 1

# (Option) You can set `--run_all True` to run all tasks, like:
python train_target.py --dset office --net resnet50 --par_consistency 0.3 --lr 1e-2 --run_all True
```

### Office-Home (Art, Clipart, Product, Real World)

```python
# Please change `--s` and `--t` to run other tasks, 
# Art -> Clipart
python train_target.py --dset office-home --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 0 --t 1

# (Option) You can set `--run_all True` to run all tasks, like:
python train_target.py --dset office-home --net resnet50 --par_consistency 0.3 --lr 1e-2 --run_all True
```

### VisDA-C (Synthetic, Real)

```python
# Synthetic -> Real
python train_target.py --dset VISDA-C --net resnet101 --par_consistency 1.0 --lr 1e-3 --sel_ratio 0.8 --s 0 --t 1
```

### DomainNet-126 (Clipart, Painting, Real, Sketch)

```python
# We evaluate 7 tasks, so please set `--s` and `--t` manually
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 0 --t 3 --das
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 1 --t 0 --das
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 1 --t 2 --das
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 2 --t 0 --das
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 2 --t 1 --das
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 2 --t 3 --das
python train_target.py --dset domainnet126 --net resnet50 --par_consistency 0.3 --lr 1e-2 --s 3 --t 1 --das
```

## Citation

If you find this code useful for your research, please cite our papers

```
# Todo
......
```



## Acknowledgement

- [SHOT](https://github.com/tim-learn/SHOT)

- [AdaContrast](https://github.com/DianCh/AdaContrast/tree/master/datasets/domainnet-126)

- [UPA](https://github.com/chenxi52/UPA)

We acknowledge their outstanding contributions!