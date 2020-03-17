
# Module 5 Final Project


## Introduction

In this lesson, we'll review all the guidelines and specifications for the final project for Module 5.

Wherever we don't specify something you can fall back on the Mod4_Project specifications.

You will be able to book a feedback session for your team's project before the final delivery of your presentation where we will focus on a particular area you would like support with.

## Final Project Summary

Congratulations! You've made it through another _intense_ module, and now you're ready to show off your newfound Machine Learning skills!

![awesome](https://raw.githubusercontent.com/learn-co-curriculum/dsc-mod-5-project/master/smart.gif)

All that remains for Module 5 is to complete the final project!

## The Project

For this project, you'll then use everything you've learned about Data Science and Machine Learning thus far to source a dataset, preprocess and explore it, and then build and interpret a classification model that answers your chosen question.
Model comparison and human reasoning are expected.

You will work with your partner to find a dataset and have it cleared with the instructors before you begin modeling.

## The data

We will be working with a dataset of your own choosing for this project! We ask that it has at LEAST 10K rows and can be used to solve a classification problem. Ideally you will have balanced classes for prediction (no trying to do outlier prediction! AKA predict the next hit song) and try to avoid models where you are doing multi-class prediction.

## The Deliverables

For online students, your completed project should contain the following four deliverables:

1. A **_Jupyter Notebook_** containing any code you've written for this project. This work will need to be pushed to your GitHub repository in order to submit your project.

2. An organized **README.md** file in the GitHub repository that describes the contents of the repository. This file should be the source of information for navigating through the repository and be an Executive Summary of your findings.

3. An **_"Executive Summary" PowerPoint Presentation_** that gives a brief overview of your problem/dataset, and each step of the CRISP-DM process.

Note: On-campus students may have different deliverables, please speak with your instructor.

### Jupyter Notebook Must-Haves

For this project, your Jupyter Notebook should meet the following specifications:

**_Organization/Code Cleanliness_**

* The notebook should be well organized, easy to follow, and code is commented where appropriate.  
    * Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code. All functions have docstrings that act as professional-quality documentation. Functions live in properly named .py files
* The notebook is written to technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.  

**_Process, Methodology, and Findings_**

* Your notebook should contain a clear record of your process and methodology for exploring and preprocessing your data, building and tuning a model, and interpreting your results.
* We recommend you use the CRISP-DM process to help organize your thoughts and stay on track.
* Don't forget to add a summary of this to your README.dm

### We will be paying attention to:
- How well you structure your CRISP-DM process in your notebook
- How well you apply validation / generalisation concepts
- How well you address your stakeholder's needs
- How well you communicate
  - Your non-technical stakeholder should know by the end of your presentation:
    - What action to take based on your analysis
    - What are the limitations of your findings (so that they don't make expensive mistakes)
  - You should be able to explain anything contained in your slides or code
- How well you implement the regression concepts we have covered in the course

## Code flow
- Data ingestion
- EDA for problem framing and guiding data cleaning
- Data cleaning + Universe definition (any row and column that you will explore at any point in your modelling)
  - In theory Feature transformations could be done here
  - Although in practice you come up with ideas after the data splitting
- Data splitting
  - X,y split
  - Train / Test split (or even Train / Validation / Test)
    - (If using Kfold CV, initialise folds now)
- Feature transformations
- Scaling (only for those classifiers that require it)
  - Fit on the training data
  - Transform on everything
- Modelling (checking performance on validation dataset)
  - Baseline model
    - Quick and easy
    - Likely to perform poorly
    - Check performance on validation data (or k-folds CV)
  - Multiple combinations of learners + features + hyper-parameter values
    - Check performance on validation data (or k-folds CV)
- Model selection
  - Comparing the validation performance across your multiple models
  - Selecting the best one for the problem that you are trying to solve
    - Interpretability counts
  - Threshold selection (only for the winning model)
    - Think carefully about the prevalence and cost that your model will meet in real life
  - Check performance on TEST dataset ONLY for your ONE chosen MODEL
- Model interpretation
  - Feature importances:
    - Which variables are most important for your model?
  - Feature impact:
    - Which variables increase/decrease your target the most?
    - By how much do the most important variables impact the target?
  - Show casing: (extra credit / not required)
    - Take a RELEVANT case (real or hypothetical) and predict the target
  - Prediction function: (extra credit / not required)
    - Create a function that
      - reverses all the data transformations
      - allows you to work exclusively with original inputs
      - provides a prediction of the target
