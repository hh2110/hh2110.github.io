---
title: "Classify MRCP Questions by Topic"
date: 2019-04-01
tags: [classification]
header:
  image: "/images/mrcp-machine/rotated.png"
excerpt: "Classification, imbalanced classes, cross validation"
mathjax: "true"
---

# Classifying MRCP Questions

Aim: classify questions by topic. 

## Summary

I applied data science techniques to classify questions by topic. 
The MRCP exam is the Member of the Royal College of Physicians Part
1 Written exam. It is sat by doctors during their training to become
a consultant in the field of medicine, as opposed to other fields, such
as Surgery, General Practice etc. 
The data-set of MCQs was created by parsing multiple pdf files taken from 
a archive of questions. 
This resulted in a data-set of 12,000 samples. 25% of the data-set 
contained unlabelled questions. After processing the questions, 
part of speech tagging was used to isolate the nouns in each sample. 
This was found to give the Random Forest classifier an F1-macro 
score of 0.79 on the test set. Techniques to combat class imbalance 
also had to be employed since the distribution of topics in the data 
set was highly skewed.

## Parsing PDFs

We used a python library here 
- Isolating questions, answers, explanations, topics
- Handling missing explanations
- Saving data

## Data Cleaning

Check the code

## Data Exploration & Visualisation

UMAP and tSNE plots

## Model Training

Decision Tree
Random Forest
Lazypredict 

### Cross Validation and Imbalanced Datasets

Plot of accuracy vs class size

## Results

79% f1 score

## Future Work\

Word embeddings: word2vec, gloVe and fastText
DNN and transfer learning.

![png](/images/mrcp-machine/topics-tsne.png)
