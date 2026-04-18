# The Geometry of Noise: Diffusion Models Without Noise Conditioning

An interactive Jupyter notebook exploring the mathematical foundations and geometric insights behind why diffusion models can learn to generate data without explicit noise conditioning. This project includes theoretical analysis, visualizations, experiments, and practical implementations.

## 📋 Table of Contents

- [Background & Motivation](#background--motivation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Notebooks Overview](#notebooks-overview)

## 🎓 Background & Motivation

Diffusion models have become a powerful class of generative models, but traditional implementations require **time/noise conditioning**: the network takes both the current state $u$ and time step $t$ as inputs, telling the model how noisy the state is.

This project explores a deeper question: **Can a single autonomous vector field $f(u)$ handle all noise levels without explicit time conditioning?**

### Key Insight

The answer is yes. While counterintuitive, a single field can learn to infer noise levels from the geometry of the input and adapt its behavior accordingly:

- **Conditioned model**: $u, t \rightarrow f(u, t)$ (explicit noise information)
- **Autonomous model**: $u \rightarrow f(u)$ (implicit inference)

The autonomous model learns the marginal expectation:
$$f^*(u) = \mathbb{E}_{t|u}[f_t(u)]$$

This means the model is not blind to noise levels—it infers them from the geometry of the input space and acts accordingly.

## 📁 Project Structure

```
geometry-of-noise-diffusion/
├── README.md                                          # This file
├── diffusion_without_noise_conditioning_notebook - colab run.ipynb
│   └── Full implementation optimized for Google Colab
├── diffusion_without_noise_conditioning_notebook- FINAL.ipynb
│   └── Final polished version with complete analysis and visualizations
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Setup Instructions

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd geometry-of-noise-diffusion
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install torch numpy matplotlib pandas jupyter scipy
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Then open either notebook file.

### Running on Google Colab

The `diffusion_without_noise_conditioning_notebook - colab run.ipynb` file is optimized for Google Colab:
1. Upload the notebook to Google Colab
2. Colab will handle dependency installation automatically
3. Run the cells sequentially

## 📖 Usage

### Quick Start

1. Open either Jupyter notebook in your environment
2. Run cells sequentially from top to bottom
3. Interactive visualizations will display inline

### What You'll Learn

- **Theoretical Foundation**: Mathematical derivation of marginal distributions and autonomous fields
- **Visualization**: Geometric views of how autonomous fields infer noise levels
- **Experiments**: Comparative analysis of conditioned vs. autonomous models
- **Extensions**: Practical applications and extensions of the core theory

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network implementation and tensor operations |
| `numpy` | Numerical computing and array operations |
| `matplotlib` | Visualization and plotting |
| `pandas` | Data manipulation and analysis |
| `scipy` | Scientific computing utilities |
| `jupyter` | Interactive notebook environment |

## 📓 Notebooks Overview

### 1. `diffusion_without_noise_conditioning_notebook - colab run.ipynb`

**Purpose**: Full implementation optimized for Google Colab execution

**Contents**:
- Dependency setup and imports
- Mathematical background on diffusion processes
- Implementation of autonomous and conditioned models
- Experimental comparisons
- Visualization of learned fields
- Extensions and practical applications

**Best For**: Running in Google Colab or exploring the full pipeline

### 2. `diffusion_without_noise_conditioning_notebook- FINAL.ipynb`

**Purpose**: Polished final version with refined analysis and comprehensive visualizations

**Contents**:
- Comprehensive mathematical derivations
- Core theoretical concepts
- Geometric visualizations of noise inference
- Detailed experimental results
- Interactive plots and animations
- Discussion of key findings
- Future directions and research extensions

**Best For**: Deep understanding of the theory and high-quality visualizations

## 🔍 Key Concepts

### Marginal Distribution
$$p(u) = \int p(u|t)p(t)dt$$

The distribution over states, averaged across all noise levels.

### Marginal Energy
$$E_{marg}(u) = -\log p(u)$$

Energy function derived from the marginal distribution, which the autonomous model implicitly learns.

### Autonomous Field
A single vector field that learns to handle all noise levels through implicit inference from geometry:
$$f^*(u) = \mathbb{E}_{t|u}[f_t(u)]$$

## 📧 Support

For questions or issues:
- Check the notebooks for detailed explanations
- Review the mathematical sections for theory
- Run experiments to verify insights

---

**Last Updated**: April 2026
