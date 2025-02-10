# TemporalMotif-CTDG
## Abstract  
TemporalMotif-CTDG is a framework for analyzing temporal graph structures using motif-based methods. The project focuses on capturing dynamic graph patterns and leveraging them for downstream tasks such as node classification and link prediction. By integrating motif-based features into a continuous-time dynamic graph (CTDG) framework, we enhance the ability to model evolving relationships within graph-structured data.
## Method Overview  
Below is **Figure 1**, which illustrates the overall workflow of our method:  

![Method Overview](images/Temporal_Graph.pdf)  
This figure illustrates the pipeline for extracting temporal motifs and incorporating them into the dynamic graph representation.

## Task Performed
We applied our framework to various temporal graph learning tasks, including node classification, link prediction, and parametric sensitivity analysis. The workflow for these tasks is shown in **Figure 2**:

![Task Workflow](images/downstream2.pdf)  
This figure highlights how motif-based features contribute to different graph learning applications.

## Repository Structure

```
TemporalMotif-CTDG/
│── CodeBase/                # Jupyter notebooks for model training and evaluation
│   ├── ablation-study.ipynb
│   ├── link_prediction.ipynb
│   ├── link_prediction_Node2Vec_50_epochs_4_models.ipynb
│   ├── node-classification.ipynb
│   ├── node_classification_Node2Vec_50_epochs_4_models.ipynb
│   ├── parametric-sensitivity.ipynb
│
│── Datasets/                # Dataset files used in experiments
│   ├── CollegeMsg.txt
│   ├── ml_enron.txt
│   ├── ml_mooc.txt
│   ├── ml_reddit.txt
│   ├── ml_wikipedia.txt
│
│── README.md                # Project documentation
│── .gitattributes            # Git attributes configuration
```

## Installation

To run the notebooks in this repository, install the required dependencies:


## Usage

The notebooks in the `CodeBase` directory provide step-by-step implementations of various experiments:

- **Link Prediction:** Evaluates the ability to predict future links in a dynamic graph.
- **Node Classification:** Classifies nodes based on learned embeddings.
- **Ablation Study:** Analyzes the impact of different parameters on model performance.
- **Parametric Sensitivity Analysis:** Studies how different hyperparameters influence results.

Each notebook is self-contained and provides instructions on running the corresponding experiments.

## Dataset

The `Datasets` directory contains real-world datasets used in experiments, including:

- **CollegeMsg:** Messaging activity among college students.
- **Enron:** Email communication network.
- **MOOC:** Online learning interactions.
- **Reddit:** Discussions from Reddit.
- **Wikipedia:** Edit interactions on Wikipedia.

## Citation

If you use this repository in your research, please cite:

```
@article{your_paper_citation,
  title={Motif Extraction and Prediction in Temporal Networks1},
  author={Alilou Marouane,Bikram Pratim Bhuyan},
  journal={Your Journal},
  year={2025}
}
