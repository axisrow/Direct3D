
# Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer (NeurIPS 2024)

![Video 2](video2.gif)
![Video 1](video1.gif)

---

## 📝 Abstract

we introduce **Direct3D**, a native 3D
generative model scalable to in-the-wild input images, without requiring a multiview diffusion model or SDS optimization. Our approach comprises two primary
components: a Direct 3D Variational Auto-Encoder **(D3D-VAE)** and a Direct 3D
Diffusion Transformer **(D3D-DiT)**. D3D-VAE efficiently encodes high-resolution
3D shapes into a compact and continuous latent triplane space. Notably, our
method directly supervises the decoded geometry using a semi-continuous surface
sampling strategy, diverging from previous methods relying on rendered images as
supervision signals. D3D-DiT models the distribution of encoded 3D latents and is
specifically designed to fuse positional information from the three feature maps of
the triplane latent, enabling a native 3D generative model scalable to large-scale 3D
datasets. Additionally, we introduce an innovative image-to-3D generation pipeline
incorporating semantic and pixel-level image conditions, allowing the model to
produce 3D shapes consistent with the provided conditional image input. Extensive
experiments demonstrate the superiority of our large-scale pre-trained Direct3D
over previous image-to-3D approaches, achieving significantly better generation
quality and generalization ability, thus establishing a new state-of-the-art for 3D
content creation.
![Framework](framework.png)

---

## 🛠️ Installation

*Coming soon...*

---

## 🚀 Usage

*Coming soon...*

---

## 📄 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{wu2024direct3d,
  title={Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer},
  author={Wu, Shuang and Lin, Youtian and Zhang, Feihu and Zeng, Yifei and Xu, Jingxi and Torr, Philip and Cao, Xun and Yao, Yao},
  journal={arXiv preprint arXiv:2405.14832},
  year={2024}
}
```

---
