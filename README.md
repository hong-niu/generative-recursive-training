# Can Generative Artificial Intelligence Survive Data Contamination? 

Official repository for the paper: _Can Generative Artificial Intelligence Survive Data Contamination? Theoretical Guarantees under Contaminated Recursive Training_ by Kevin Wang, Hongqian Niu, and Didong Li (2026). 

Arxiv: https://arxiv.org/abs/2602.16065

This repository contains the Python and notebook implementations required to reproduce the simulations and empirical experiments presented in the paper. 

## Overview

As generative AI models (like LLMs and diffusion models) increasingly populate the internet with synthetic data, future models are inevitably trained on a mixture of human-generated and AI-generated content. Previous work studying this process in simplified settings (discrete or Gaussian data) concluded that it inevitably leads to model collapse. 

This work fills a critical gap by studying contaminated recursive training in a generalized framework, allowing the generative model to act as a universal approximator with minimal assumptions on the true data distribution.

**Key Contributions:**
- Proof of Convergence: We theoretically prove that contaminated recursive training can still converge, avoiding model collapse. 
- Convergence Rate: We theoretically establish the convergence rate is the minimum of the baseline model's convergence rate and the fraction of real data used in each iteration.
- Biased Sampling: We extend the theoretical guarantees to realistic scenarios where sampling bias is present during real data collection.
- Empirical Validation: Extended simulations using KDE/ECDF and WGAN estimators, and more realistic diffusion and LLM experiments. 

## Repository Structure

The codebase is organized into two primary directories: 

```text
generative-recursive-training/
├── experiments/       # Code for empirical experiments on real-world datasets/models
├── simulations/       # Code and notebooks to reproduce theoretical simulations 
├── README.md          # Project overview and setup instructions

The scripts and notebooks are mostly self-contained, see manuscript for experimental parameters used. Manuscript figures are generated via separate notebooks. 
