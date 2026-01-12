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
## Tasks Performed  

The framework supports the following temporal graph learning tasks:

- **Node Classification**
- **Link Prediction**
- **Ablation Studies**
- **Parametric Sensitivity Analysis**

These tasks are designed to evaluate the contribution of temporal bias, motif sets, incidence matrices, and temporal/structural encoders within the CTDG pipeline.

---

## Repository Structure  

## Repository Structure

```
TemporalMotif-CTDG/
│── configs/                # Jupyter notebooks for model training and evaluation
│   ├── protocol.yaml
│── data/                # Dataset files used in experiments
│   ├── ml_CollegeMsg.csv
│   ├── ml_enron.csv
│   ├── ml_mooc.csv
│   ├── ml_reddit.csv
│   ├── ml_wikipedia.csv
│
├── images
├── runners
    ├── ablations_ssm_encoder.py
    ├── run_nodecls.py
    ├── run_nodescls_primary.py
    ├── run_ours.py
├── scripts
    ├──ablation_incidence_only_tuned.py
    ├──....
├── src
    ├── dataio
    ├── sampling
    ├── temporal
├── tools
│── README.md                # Project documentation
│── .gitattributes            # Git attributes configuration
```

---

## Installation

To run the notebooks in this repository, install the required dependencies:

## Usage

The scripts in the runners/ and scripts/ directories provide step-by-step execution of experiments:

### Main Method :

python runners/run_ours.py --config configs/protocol.yaml

### Node Classification:

python runners/run_nodecls.py

### Ablation Studies:

python scripts/run_ablations.py


### Parametric Sensitivity Analysis:

python scripts/run_param_sensitivity.py


## Dataset

The `Datasets` directory contains real-world datasets used in experiments, including:

- **CollegeMsg:** Messaging activity among college students.
- **Enron:** Email communication network.
- **MOOC:** Online learning interactions.
- **Reddit:** Discussions from Reddit.
- **Wikipedia:** Edit interactions on Wikipedia.
