# Applied Deep Learning Project Instructions

**Course:** 60.001 Applied Deep Learning, Y2026

---

## 1. Introduction

You may work on a topic of your own choice, provided it fulfils the requirements detailed below. You will design a custom model and either create your own dataset or source one online (e.g. [Kaggle](https://www.kaggle.com/), Google Dataset Search).

> Do not waste too much time on creating or processing a dataset -- this is not the purpose of this project.

Treat this project as a professional delivery to a client (the professors). Pay attention to code quality and documentation.

**Submission:** Upload code and notebooks to eDimension -- **do not include the dataset**. Upload your final project to GitHub for your portfolio. If the dataset is required to run the project, provide a download link (Kaggle, Google Drive, or Dropbox).

**Presentation:** Week 13 session (date TBC).

---

## 2. Key Details

| Item | Detail |
|------|--------|
| **Team size** | 3--4 people (solo work not allowed) |
| **Theme** | AI for Social Good |
| **Proposal deadline** | March 6th, 11:59 PM (submit via email to pritee_agrawal@sutd.edu.sg) |
| **Submission deadline** | April 17th, 11:59 PM |
| **GPU access** | SUTD AI Mega Cluster (see instructions manual) |
| **Cluster login** | Username: `studentID`, Password: `AIcluster#2025` (change after first login) |
| **Group formation** | [Google Sheet](https://docs.google.com/spreadsheets/d/1neYErW2lhWJiHm1COvZ1opODIZXvINxZ2QHUxj11FI/edit?usp=sharing) |

---

## 3. Grading Rubrics

### Technical Implementation (50%)

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Concept & Relevance** | 5% | Thorough understanding of relevant concepts and techniques demonstrated in the implementation |
| **Coding** | 25% | Well-structured, efficient, and thoroughly documented code; reproducibility |
| **Performance & Evaluation** | 20% | Hyperparameter tuning, comparison to baselines, appropriate metrics |

**Required metrics by task type:**

- Classification: Accuracy, Precision-Recall, F1-score
- Regression: RMSE, MAE
- Generative models: FID, Inception Score

### Presentation and Communication (20%)

- Clear and well-organized presentation
- Effective communication of objectives, methodology, and results
- Demonstration of impact or potential value

### Project Report (30%)

- **Introduction:** Problem statement with objectives, scope, and constraints
- **Method:** Architecture and model description; include failed approaches to demonstrate effort
- **Experiments:** Specifications and comparison of results
- **GitHub link:** Public repository with code on eDimension and GitHub; dataset/weights on Google Drive/Dropbox if heavy
- **Conclusion**

### Creativity and Innovation (5% bonus)

- Creative solutions or innovative approaches beyond state-of-the-art
- Unique features that add significant value

---

## 4. Project Proposal Guidelines

Submit a brief PDF via email to pritee_agrawal@sutd.edu.sg by **March 6th, 11:59 PM**. Include:

1. **Topic** and problem to be investigated
2. **Expected inputs and outputs**, dataset to be used
3. **Architecture draft** (type of architecture planned -- does not need to be final)
4. **Team members**
5. **Deliverables** -- at minimum:
   - PDF report
   - Code for training the model from scratch
   - Code for recreating the trained model from a saved file

### Theme Requirement

The model must align with **"AI for Social Good"** -- addressing a real-world societal challenge that benefits communities, the environment, or humanity.

**Example domains:**

- **Healthcare:** Chest X-ray classification (pneumonia/TB), wearable health monitoring, clinical text summarization
- **Education & Accessibility:** Handwritten equation solver (CNN + OCR), AI text-to-speech for visually impaired
- **Environmental Sustainability:** Air pollution prediction, wildlife poaching detection
- **Misinformation Detection:** Fake news detection (BERT), cyberbullying detection, deepfake video detection, hate speech classification

---

## 5. Project Expectations

### Training & Evaluation

- Train on a training set, evaluate on a **separate test set**
- Use a **train/validation/test split** (preferred)
- Perform **hyperparameter tuning**
- Compare performance against **state-of-the-art baselines** (you are not expected to beat them)

### Reproducibility

- Save model weights to a file
- Describe all steps to recreate the architecture from scratch
- Provide clear instructions to retrain and to load saved weights for evaluation
- List all package dependencies

### Visualization & Analysis

- Include accuracy/loss curves and performance visualizations
- For every figure in the report, describe how to recreate it
- Show examples of model failures and discuss potential causes

### Documentation

- Report should contain everything needed to run the code
- Include group members and their individual contributions

---

## 6. Delivery

### Report

- PDF format, submitted with code
- Code may be `.py` files, `.ipynb` notebooks, or Google Colab notebooks
- Explicitly state how to run the code (imports, dataset location, commands)

**Recommended format:**

- Jupyter Notebook combining Markdown and code cells
- Separate `.py` files for larger code components, imported into the notebook

### GitHub Repository

Your repository should contain:

- PDF report
- Documented code/notebook files
- Directions for required libraries and retraining from scratch
- Instructions to recreate the trained model by loading saved weights
- Link to dataset/weights on Google Drive/Dropbox if too large for GitHub

> You are nearing the end of your curriculum -- consider starting your project portfolio on GitHub if you haven't already.

### Presentation

- Week 13 session (date TBC)
- Include a demo, slides, or short video
