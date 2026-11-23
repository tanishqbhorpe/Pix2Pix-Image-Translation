# Pix2Pix – Image-to-Image Translation using Conditional GAN

An image-to-image translation project using a **Conditional Generative Adversarial Network (cGAN)** based on the Pix2Pix architecture and implemented with PyTorch.

This project converts semantic segmentation maps into realistic building façade photographs using a U-Net Generator and PatchGAN Discriminator.

Developed as part of the **Prodigy InfoTech Generative AI Internship – Task 4**.

## Project Overview

Pix2Pix is a conditional GAN framework that learns a mapping from an input image to a corresponding target image.

In this project:

- **Input:** Building semantic segmentation map
- **Output:** Generated building photograph
- **Dataset:** Berkeley Facades Dataset
- **Framework:** PyTorch
- **Training environment:** Google Colab GPU

## Project Architecture

### 1. Generator – U-Net

The Generator converts segmentation maps into building photographs.

Features:
- Encoder-decoder architecture
- Eight encoder blocks
- Seven decoder blocks
- Skip connections
- Tanh output activation

### 2. Discriminator – PatchGAN

The Discriminator evaluates whether an input-target image pair is real or generated.

Features:
- Conditional image-pair input
- Patch-based classification
- 70 × 70 PatchGAN design
- BCEWithLogitsLoss

## Technologies Used

- Python
- PyTorch
- Conditional GAN (cGAN)
- Pix2Pix
- NumPy
- Matplotlib
- Pillow
- Google Colab

## Dataset

**Berkeley Facades Dataset**

The dataset contains paired images of building photographs and their corresponding semantic segmentation maps.

| Dataset split | Images |
|---|---:|
| Training | 400 |
| Validation | 100 |
| Testing | 106 |
| **Total** | **606** |

Dataset source: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz

## Training Configuration

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 1 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| L1 loss weight | 100 |
| Epochs completed | 40 |
| Hardware | Google Colab GPU |

### Loss Function

The Generator uses adversarial loss combined with L1 reconstruction loss.

Generator objective:

`L_G = L_GAN + 100 × L_L1`

The adversarial loss encourages realistic image generation, while L1 loss encourages similarity to the corresponding target photograph.

## Experimental Results

The model was trained for 40 epochs with validation monitoring and checkpoint saving.

### Best validation result

| Metric | Value |
|---|---:|
| Best validation L1 | 0.3573 |
| Best checkpoint | Saved |

### Test evaluation

The final evaluation was conducted on all 106 test images.

| Metric | Result |
|---|---:|
| Test images | 106 |
| Average L1 | 0.3664 |
| Average MSE | 0.0570 |
| Average PSNR | 12.66 dB |

These metrics measure pixel-level similarity and do not directly measure perceptual realism.

## Sample Results

The repository includes a sample comparison showing:

1. Input segmentation map
2. Pix2Pix generated image
3. Ground-truth building photograph

See `results/sample_comparison.png`.

## Trained Model Downloads

The trained PyTorch model files are hosted separately on Google Drive because of their size.

**[Download Trained Models – Google Drive](https://drive.google.com/drive/folders/1n1ITOjMhP1reGSwRS2fzYY9MclP7_HRS?usp=drive_link)**

Available files:

- `best_generator.pth` – Best-performing Generator checkpoint.
- `latest_checkpoint.pth` – Latest training checkpoint, including Generator, Discriminator, and optimizer states.

Access requires the Google Drive folder to be shared with anyone who has the link.

## Running the Project

### 1. Open the notebook

Open `Pix2Pix.ipynb` using Google Colab.

### 2. Enable GPU

Select:

Runtime → Change runtime type → T4 GPU (if available).

### 3. Install dependencies

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

### 4. Download the dataset

Run the dataset download and extraction cells in the notebook.

### 5. Execute the notebook

Run the notebook cells in sequence to preprocess the data, initialize the networks, train the model, and evaluate the generated images.

### 6. Load the trained Generator

Download `best_generator.pth` from Google Drive and load it into the Generator architecture defined in the notebook.

## Limitations

- Training used a relatively small dataset.
- Generated images still contain noticeable texture noise.
- Fine architectural details are not always accurately reproduced.
- Pixel-based metrics do not fully capture visual realism.
- Results depend on the training data and model configuration.

## Future Improvements

- Train for more epochs with careful validation monitoring.
- Experiment with learning-rate scheduling.
- Improve image preprocessing and augmentation.
- Explore perceptual loss and additional evaluation metrics.
- Test on larger image datasets.
- Build an interactive image translation interface.

## Author

**Tanishq Bhorpe**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: https://github.com/tanishqbhorpe

LinkedIn: https://www.linkedin.com/in/tanishq-bhorpe

---

*Developed as part of the Prodigy InfoTech Generative AI Internship – Task 4.*
