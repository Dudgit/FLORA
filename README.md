# CT to RSP Conversion with Deep Learning: A Latent Flow-Matching Approach

**Anonymous Submission — MICCAI SASHIMI 2026**


This repository contains the reference implementation for our SASHIMI 2026 workshop submission. The project introduces a two-stage deep learning framework that performs continuous transformation from X-ray Computed Tomography (CT) into Relative Stopping Power (RSP) volumes entirely within a compressed latent space.

*Note: This codebase is currently under active development. The scripts provided here are intended for peer-review transparency and methodology verification.*

## Project Overview

Accurate estimation of RSP is critical for proton therapy treatment planning. Current clinical standards rely on stoichiometric lookup tables (HLUT), which introduce range uncertainties, particularly in highly heterogeneous, low-density regions like lung tissue. 

This project bypasses voxel-space translation by modeling the CT-to-RSP relationship as a continuous transport problem in a latent manifold.

### The Pipeline

The architecture consists of two primary stages:

1. **Stage I: Latent Representation Learning (VAE-GAN)**
   Inspired by the MAISI architecture, we train a high-capacity 3D autoencoder to compress volumetric CT and RSP data into a dense, high-quality latent representation. This preserves structural fidelity while significantly reducing computational overhead.
2. **Stage II: Latent Flow Matching**
   With frozen autoencoder weights, a conditional flow-matching network learns the deterministic velocity field to transport a CT latent state to its corresponding RSP latent state. This enables efficient inference and naturally supports the integration of auxiliary conditional data (e.g., proton detector measurements). The final goal is to do include this to obtain the highest possible accuracy and the best possible dosage delivery.

##  Dataset

A major bottleneck in CT-to-RSP modeling is the lack of paired in-vivo ground truth. To address this, we constructed a biologically plausible dataset using the public CT-RATE dataset and Monte Carlo simulations (OpenGATE). For submission we used 36 increments for every slice, yet we are constructing a much better resolution. Due to the simulation time and only-CPU implementation, this takes a huge amount of time.
