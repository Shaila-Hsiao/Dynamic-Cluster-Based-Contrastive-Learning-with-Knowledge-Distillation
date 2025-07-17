## 📘 Dynamic Centroid-Based Contrastive Learning with Knowledge Distillation

<img src="./img/DCBCL_framework.png" width="600">

---

### 📦 Requirements

- ✅ Python ≥ **3.10**
- ✅ PyTorch ≥ **2.5** (with CUDA 11.8)
- ✅ Dataset: [Tiny-ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
- ✅ Install dependencies:

```bash
# Install faiss-gpu (for clustering)
conda install -c conda-forge faiss-gpu

# Install tqdm, PyTorch with CUDA 11.8 support
pip install tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````
#### Optional
- wandb
```bash
pip install wandb
```
---

### 🗂️ Dataset Setup (Tiny-ImageNet)

1. Edit the dataset root in `tiny.py`:

```python
tiny_imagenet_root = '[Tiny-ImageNet dataset folder]'
```

2. Reorganize validation structure:

```bash
python tiny.py
```

---

### 📥 Download Pre-trained Teacher (MoCo)

[Download Link (Google Drive)](https://drive.google.com/file/d/1JZ5YX6AUukPm8hB2RWMCgW0MUABG6650/view?usp=drive_link)

---

### 🧪 Pre-Training

```bash
python main_dcbcl_resnet.py \
  -a resnet50 \
  --lr 0.05 \
  --batch-size 256 \
  --temperature 0.05 \
  --mlp \
  --aug-plus \
  --cos \
  --proportion 0.2 \
  --alpha 0.2 \
  --dataset TinyImageNet \
  --exp-dir [your_output_dir] \
  --pretrained [path_to_teacher_checkpoint] \
  --student-ratio 20% \
  --use-kd \
  --use-centroid \
  --use-masking \
  [Tiny-ImageNet dataset folder]
```

> 💡 Use `--mlp`, `--aug-plus`, and `--cos` for PCL v2-style setup.

---

### 🎯 Linear Evaluation

```bash
python eval_cls_imagenet_ratio.py \
  --pretrained [path_to_student_model] \
  -a resnet50 \
  --lr 0.01 \
  --batch-size 256 \
  --epochs 200 \
  --patience 20 \
  --student-ratio 20% \
  --dataset TinyImageNet \
  --exp-dir [your_eval_output_dir] \
  [Tiny-ImageNet dataset folder]
```
