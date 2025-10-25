# CandleGen: Synthetic OHLC Data Generator using GANs

**CandleGen** is a synthetic **OHLC (Open-High-Low-Close)** M1 data generator designed to simulate various market trends using **Generative Adversarial Networks (GANs)**.
It supports the generation of synthetic candlestick data for the following market conditions:

- **Strong Bull**
- **Bull**
- **Flat**
- **Bear**
- **Strong Bear**

---

## Repository Overview

This repository provides:

- Preprocessing scripts
- GAN-based models for each market trend
- Evaluation utilities to assess the quality of generated data

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/KMKyaw/CandleGen.git
cd CandleGen
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

> **Note:** The project was developed and tested with **Python 3.10**.

---

## 📁 Project Structure

```
CandleGen/
│
├── prepare.py                # Preprocessing code for raw OHLC data
│
├── bullx2/                   # Generator for strong bull market data
├── bull/                     # Generator for bull market data
├── flat/                     # Generator for flat market data
├── bear/                     # Generator for bear market data
├── bearx2/                   # Generator for strong bear market data
│
└── Evaluate/                 # Evaluation scripts for generator performance
```

Each trend folder (e.g., `bull/`, `bear/`, etc.) contains the GAN model and scripts required to generate synthetic data specific to that market condition.

---

## Evaluation

The `Evaluate/` folder contains scripts to assess the performance and realism of each generator, comparing generated data against real market distributions.
