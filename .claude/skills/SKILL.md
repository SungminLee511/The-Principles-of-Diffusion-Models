# Diffusion Model Theory Skill

> **MAINTENANCE:** This file and all sub-files in `.claude/skills/` are living documents. When you modify code in this project, update the relevant skill file in the **same commit**. Added a module? Update the file tree. Changed a function signature? Update the docs. New gotcha? Add it.

**Trigger:** Use this skill when the user asks about foundational diffusion model theory — VAEs, DDPM, NCSN, score-based generative models, variational bounds, noise conditional score networks, or the SDE unification framework. Also activate when working on educational notebooks in this repository.

**Project Root:** `/home/RESEARCH/The-Principles-of-Diffusion-Models/`
**Reference Paper:** https://www.arxiv.org/abs/2510.21890

---

## Repository Structure

```
The-Principles-of-Diffusion-Models/
├── 0.PDF/                              # Reference papers (read alongside notebooks)
│   ├── 2. Variational Perspective from VAEs to DDPMs.pdf
│   └── 3. Score based perspective From EBMs to NCSNs.pdf
├── 2.Variational_Perspective/
│   ├── VAE.ipynb                       # Variational AutoEncoder from scratch
│   └── DDPM.ipynb                      # Denoising Diffusion Probabilistic Models
├── 3.Score_Based_Perspective/
│   └── NCSN.ipynb                      # Noise Conditional Score Networks
└── README.md
```

## Learning Progression

### 1. VAE — Variational AutoEncoder
- Encoder q(z|x) maps data to latent space
- Decoder p(x|z) reconstructs from latent
- ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
- Key idea: reparameterization trick for backprop through sampling

### 2. DDPM — Denoising Diffusion Probabilistic Models
- Forward process: gradually add Gaussian noise over T steps (fixed)
- Reverse process: learn to denoise step-by-step (neural network)
- Loss: simplified to ‖ε - ε_θ(x_t, t)‖² (predict the noise)
- Connection to VAE: DDPM = hierarchical VAE with T latent layers and fixed encoder

### 3. NCSN — Noise Conditional Score Networks
- Learn the score function: s_θ(x) ≈ ∇_x log p(x)
- Score matching objective: E[‖s_θ(x) - ∇_x log p(x)‖²]
- Annealed Langevin dynamics for sampling
- Connection to DDPM: denoising objective ≡ score matching at each noise level

### Theoretical Unification (SDE Framework)
- DDPM → VP-SDE (Variance Preserving)
- NCSN → VE-SDE (Variance Exploding)
- Both are discretizations of continuous-time SDEs
- Reverse-time SDE depends on the score function ∇log p_t(x)

## Notebook Execution

Follow the global notebook protocol — NEVER run `.ipynb` directly:
```bash
conda run -n SML_env nohup python -u <notebook_as_script>.py > <log>.txt 2>&1 &
```

## Gotchas

1. **Notebook Dependencies:** Educational notebooks may have inline `pip install` cells. Check for version conflicts with `SML_env` before running.
2. **PDF References:** The PDFs in `0.PDF/` are the theoretical foundation. Always read them alongside the notebooks for full context.
3. **Dataset Downloads:** DDPM and NCSN notebooks likely download datasets (MNIST/CIFAR). Check disk space before running.
4. **GPU Memory:** DDPM training is memory-hungry even on small datasets. Check `nvidia-smi` first.
5. **Chapter Numbering:** Chapters start at 2 (variational) and 3 (score-based). Chapter 1 (introduction/background) is covered in the arxiv paper but has no notebook yet.
6. **This is a study repo:** Pure educational — no production code, no tests, no CI. Changes should maintain pedagogical clarity over code efficiency.
