# Budget-constrained Collaborative Renewable Energy Forecasting Market

This repository contains the core implementations and datasets used in our paper:

> **"Budget-constrained Collaborative Renewable Energy Forecasting Market"**  
> *Carla Gonçalves, Ricardo J. Bessa, Tiago Teixeira, João Vinagre*  
> *Published in IEEE Transactions on Sustainable Energy, 2025*

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project structure](#2-project-structure)
3. [Running the code](#3-dependencies)
4. [License](#4-license)
5. [Citation and contact](#5-citation)


## 1. Introduction  
This project presents a **budget-constrained collaborative forecasting market**, where data owners can **monetize their features**, and buyers can **optimize forecasting performance while staying within a budget**.

### 1.1. What is the main purpose of this project?
This project implements a **collaborative forecasting market** that allows data sellers to monetize their data while data buyers purchase relevant features under a given budget constraint. It uses **Spline LASSO regression** to improve the accuracy of renewable energy forecasting (e.g., wind or solar).

 - **Collaborative forecasting model**: **Spline LASSO regression** 
 - **Bid-based data market** where:
   - **Data sellers** set prices for their features.  
   - **Data buyers** specify a budget or price related to forecast accuracy improvements.  
 - **Incentive Mechanism**: Estimates the more accurate LASSO B-spline model within the budget 
 

### 1.2. Which forecasting tasks does this code target?
We primarily target **wind power forecasting**, though the underlying models can be adapted to other use cases (solar, load forecasting, etc.) if you have suitable time series data.

### 1.3. Where can I find the related paper?

🔗 **Read the full paper**: [ArXiv](https://arxiv.org/pdf/2501.12367?) or  [IEEE](https://ieeexplore.ieee.org/abstract/document/10850726?casa_token=ropqQfr9yCoAAAAA:yxskVfrxzZJDybn6PPwt-b-BzCLdFQFzAXuK0Gvg-blXnXoV_-bjOA36-sefOM3dRFPkZ1WO)

---

## 2. Repository Structure

The codes provided are organized as follows:

```
├── data/                                          # Datasets used in experiments
├── src/                                           # Main implementation files
│   ├── proposal.py                                # Algorithmic solution 
│   └── custom_pipeline.py                         # Feature selection filters
├── examples/                                      
│   ├── 1-a-generate-basic-synthetic-data.py       # Basic synthetic setup 
│   ├── 1-b-run-markets-basic-synthetic-data.py    # Run the proposal
│   ├── 2-advanced-synthetic-run-data-markets.py   # Advanced synthetic setup
│   ├── 3-a-gefcom2014-create-data-folders.py      # Wind power setup (GEFCom 2014) 
│   ├── 3-b-gefcom2014-compare-models.py           
│   ├── 3-b-gefcom2014-draw-plots.py               
│   └── 3-c-gefcom2014-run-data-market.py          
├── requirements.txt        # Dependencies
└── README.md               # This file
```

### 2.1. Are the datasets fully included in the repository?
- **Synthetic Datasets**: Example synthetic data is stored in the `data/` folder, generated via scripts in the `examples/` directory.  
- **Real-world Datasets**: We provide scripts to process the **GEFCom2014** wind power dataset. You need to **manually download** the original GEFCom2014 ([link here](https://www.dropbox.com/scl/fi/zwnlfwhds3k2xz0/GEFCom2014.zip?rlkey=oz7unehbwtglp1pbgbnb4wne2&e=1&dl=0)).

---

## 3. Running the Code

###  Clone the repository
```bash
git clone https://github.com/INESCTEC/budget-constrained-collaborative-forecasting-market.git
cd budget-constrained-collaborative-forecasting-market
```

### Install dependencies
```bash
pip install -r requirements.txt
```

Dependencies include:
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-optimize`
- `scipy`
- `sympy`
- `plotnine`

---

The results can be obtained by running the scripts in the 'examples' folder, e.g.:

#### Generate the basic synthetic dataset
```bash
python examples/1-a-generate-basic-synthetic-data.py
```

#### Run the market and obtain the results in Table I of the paper

A folder `results/` will be created to save a .csv with Table I results.
```bash
python examples/1-b-run-markets-basic-synthetic-data.py
```

## 5. License

This project is licensed under the AGPL v3 license - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer: ** This code contains parts adapted from the original implementation provided in [Yu, G., Fu, H., & Liu, Y. (2022). High-dimensional cost-constrained regression via nonconvex optimization. Technometrics, 64(1), 52-64.](https://www.tandfonline.com/doi/full/10.1080/00401706.2021.1905071) which was published and released as open-source by the authors.  This version contains modifications, improvements, and deviations from the original code.

---

## 6. Citation and contact

If you use this code, please cite our work:

```
@article{goncalves2025budget,
  title={Budget-constrained Collaborative Renewable Energy Forecasting Market},
  author={Gon{\c{c}}alves, Carla and Bessa, Ricardo J and Teixeira, Tiago and Vinagre, Jo{\~a}o},
  journal={IEEE Transactions on Sustainable Energy},
  year={2025},
  publisher={IEEE}
}
```

---

For questions or collaborations, feel free to reach out:

- **Carla Gonçalves** - [carla.s.goncalves@inesctec.pt](mailto:carla.s.goncalves@inesctec.pt)
- **Ricardo J. Bessa** - [ricardo.j.bessa@inesctec.pt](mailto:ricardo.j.bessa@inesctec.pt)

---
