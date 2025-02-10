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
###CodeBase/ – Contains Jupyter notebooks for experiments:

ablation-study.ipynb – Ablation study experiments
link_prediction.ipynb – Link prediction experiments
link_prediction_Node2Vec_50_epochs_4_models.ipynb – Link prediction using Node2Vec
node-classification.ipynb – Node classification experiments
node_classification_Node2Vec_50_epochs_4_models.ipynb – Node classification with Node2Vec
parametric-sensitivity.ipynb – Sensitivity analysis on model parameters

###Datasets/ – Includes various temporal graph datasets:

CollegeMsg.txt – College messaging dataset
ml_enron.txt – Enron email dataset
ml_mooc.txt – MOOC interaction dataset
ml_reddit.txt – Reddit discussions dataset
ml_wikipedia.txt – Wikipedia edits dataset
