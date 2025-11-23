# **DeepHalo: Replication of Featureless Deep Choice Model (Zhang et al., 2025\)**

## **🎯 Project Goal**

This project aims to replicate the **featureless** DeepHalo model, a deep neural network-based Discrete Choice Model (DCM), as introduced by Zhang et al. (2025). The core objective is to validate the effectiveness of the proposed architecture and to reproduce its reported performance in terms of Approximation Error (RMSE).

## **📂 File Structure and Descriptions**

This repository contains the necessary files for synthetic data generation, model training, quality assurance, and key project documentation.

| Filename | Type | Description |
| :---- | :---- | :---- |
| synthetic\_data\_deephalo.ipynb | Jupyter Notebook | **Full Experiment Script (9 Cells).** Contains the complete code for data generation, model construction, unit tests, and the full 500-epoch training run. |
| synthetic\_data\_deephalo\_epoch100.ipynb | Jupyter Notebook | **Rapid Verification Script.** A simplified version for fast testing and stability checks, running a total of 100 epochs with validation performed every 50 epochs. |
| testing\_plan.md | Document (Markdown) | **Formal Testing Plan.** Outlines the project's Quality Assurance strategy, defining the scope, validation targets (mathematical constraints, data pipeline, and architecture), and final acceptance criteria. |
| testing\_report.md | Document (Markdown) | **Formal Testing Report.** Summarizes the outcomes of all Unit Tests and Integration Tests. Confirms the code's stability and correctness, validated on the **Google Colab (T4 GPU)** environment. |
| report.pdf | Document (PDF) | **Project Research Report.** A comprehensive summary document covering the DeepHalo methodology, a detailed discussion of known implementation issues, and a review of relevant literature. |

## **✅ Testing and Validation Status**

All core components have been rigorously validated through unit and integration testing:

* **Mathematical Constraints:** The parameter budget solving logic (calculate\_layer\_width function) is confirmed to be mathematically sound.  
* **Data Pipeline:** Data generation and caching (.npz files) mechanisms are stable and verified for integrity.  
* **Model Architecture:** The critical **Quadratic Activation Layer** and **Availability Masking Layer** are confirmed to be correctly implemented.

For detailed results, please refer to the testing\_report.md.

## **⚙️ Running the Project**

1. **Environment Setup:** Ensure your environment (Google Colab recommended) has the necessary libraries installed: TensorFlow/Keras, NumPy, and Matplotlib.  
2. **Data Generation:** Run synthetic\_data\_deephalo.ipynb first to generate and cache the required synthetic dataset.  
3. **Model Training:** Use the appropriate Jupyter Notebook to initiate the full-scale experiment based on the featureless DeepHalo architecture.