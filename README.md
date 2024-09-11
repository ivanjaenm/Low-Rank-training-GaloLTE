## Poster Presentation

<img width="961" alt="image" src="https://github.com/user-attachments/assets/030ba5ae-fdde-4204-a694-d95fbaaab868">

# Lowrank Training 

Low-rank adaptation (LoRA) was originally proposed for fine-tuning by reparameterizing weight with its factorization of two lower-dimensional matrices A and B. In pre-training, low-rank adaptation does not achieve performance similar to full parameter training. Recent works have extended LoRA to improve its performance. LoRA-The-Explorer (LTE) suggested tranining multiple LoRA heads in parallel. GaLore projects gradients into lower-dimensional matrices and updates weights in a smaller subspace. In this project, we have applied LTE and GaLore methods in foundation models for language and vision tasks to validate their effectiveness. Our key results are as follows-
* Using multiple heads with a small rank can perform same as or slightly better than full-parameter training of GPT model.
* Fine-tuning vision transformers (ViTs) with LoRA acheives better accuracy with less number of epochs compared to full parameter fine-tuning. 
* Applying LoRA on attention layers is most effective and a good balance between model performance and number of trainable parameters. 

## LTE tasks

Sources: 
* [nanoGPT](https://github.com/karpathy/nanoGPT)
* [nanoGPT-LoRA](https://github.com/danielgrittner/nanoGPT-LoRA)
* [LTE](https://github.com/minyoungg/LTE)

Initial setup (also see source repo):
1. run python3 data/cnn_dailymail/prepare.py
2. python3 train_gpt_lte.py

## GaLore tasks

Sources:
*  [GaLore](https://github.com/jiaweizzhao/GaLore)

## LoRA Vision

Sources:
* [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)

Dataset: 
* [MedMNIST](https://medmnist.com/) 

Source code:
* `meloravit.py`: LoRA-ViT + MeLo + GaLore optimizer
* `ltegalorevit.py`: Combined version of lte and galore that we got mostly working.

## Contributors

* [Iván Jaen Márquez](https://github.com/ivanjaenm)
* [Laik Ruetten](https://github.com/ivanjaenm)
* [Sadman Sakib](https://github.com/sadmankiba)
* [Zheyang Xiong](https://github.com/zxiong44)

## References 

1. LoRA: Low-rank adaptation of large language models. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, W. Chen, and T.-Y. Liu. arXiv preprint arXiv:2106.09685, 2021.
2. Training neural networks from scratch with parallel low-rank adapters. M. Huh, B. Cheung, J. Bernstein, P. Isola, and P. Agrawal. arXiv preprint arXiv:2402.16828, 2024.
3. Galore: Memory-efficient LLM training by gradient low-rank projection. J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian. arXiv preprint arXiv:2403.03507, 2024.
