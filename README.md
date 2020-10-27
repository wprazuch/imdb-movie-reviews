### IMDb Movie Reviews

This repository contains some methods, analysis, experiments and deployment strategies for a common task of sentiment analysis from a well-known IMDb dataset.
The goal of the project is to present general approach for dealing with a Machine Learning project. Although the task of sentiment analysis from IMDb is simple, some intermediate
concepts were shown - mainly for explainability, EDA, and serving/deployment. The whole project can be run in Docker container.

## Data
A well-known dataset of movie reviews from IMDb was used in this task. It can be downloaded by using common API from libraries such as Tensorflow, or PyTorch.

## Methods
Couple of methods were presented in the project. First, some common Machine Learning algorithms are used in sentiment assesment on the dataset. It is clear that simple
machine learning algorithms do well on this problem, but deep learning models were used as well. Some EDA and Explainability methods were added to the repository.

For deployment, KFServing was used for `sklearn` models.

Serving of PyTorch models is under development and hopefully will be finished soon.
