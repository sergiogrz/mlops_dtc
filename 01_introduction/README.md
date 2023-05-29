# Module 1 - Introduction

## Table of contents

1. [Introduction to MLOps](#1-introduction-to-mlops).
2. [Environment preparation](#2-environment-preparation).
3. [Training a model from scratch](#3-training-a-model-from-scratch).
4. [Course overview](#4-course-overview).
5. [MLOps maturity model](#5-mlops-maturity-model).



## 1. Introduction to MLOps

[Video source](https://www.youtube.com/watch?v=s0uaFZSzwfI).

MLOps (Machine Learning Operations) is a paradigm which consists of a set of best practices for putting Machine Learning into production.

A Machine Learning project can be simplified to 3 steps:
1. **Design**
    + Do we really need ML to solve the problem we are interested in? Or there is something simpler we can use?
    + If we need ML, we go to the next stage.
1. **Train**
    + We try to find the best possible model, by doing and evaluating different experiments.
    + As a result, we have a model that we will apply on new data.
1. **Operate**
    + Model deployment, management and monitoring.

MLOps helps us in all these 3 stages.


The example problem considered throughout the course aims to predict the duration of a taxi trip.


## 2. Environment preparation

[Video source](https://www.youtube.com/watch?v=IXSiYkP23zo).

The video shows the steps to set up a Linux VM in AWS ([via Amazon EC2](https://aws.amazon.com/getting-started/launch-a-virtual-machine-B-0/)). In my case, I use my local Linux machine to follow the course.

System requirements and versions used througout the course:
* Python (installed via [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)).
* [Docker](https://docs.docker.com/engine/install/ubuntu/).
* [Docker Compose](https://docs.docker.com/compose/install/).

```bash
Python 3.9.16
Docker 23.0.6
Docker Compose 2.12.2
```

Python packages required are listed in [environment.yml](../environment.yml).



## 3. Training a model from scratch

[Video source](https://www.youtube.com/watch?v=iRunifGSHFc).

In [this notebook](./trip_duration_prediction.ipynb) we download data from [NYC TLC Trip Record website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and we train a simple model for predicting the duration of a taxi trip.



## 4. Course overview

[Video source](https://www.youtube.com/watch?v=teP9KWkP6SM).

Performing Machine Learning experiments solely on Jupyter Notebook has limitations in terms of reproducibility, code organization, experiment management, collaboration, and scalability. To overcome these challenges, using experiment tracking tools is recommended. These tools provide feautres like version control, parameter tracking, metric logging, artifact management, and collaboration. Additionally, they often include a model registry, which is essential for tracking and managing trained models throughout their lifecycle. The model registry ensures model versioning, easy model deployment, and model lineage, enhancing reproducibility and facilitating model governance and deployment processes.

Module 2 of the course covers this, by using tools like [MLflow](https://mlflow.org/) for experiment tracking and model registry.

Module 3 covers orchestration and ML pipelines, their main benefits and best practices. We make use of [Prefect](https://www.prefect.io/) as orchestration tool.

Module 4 introduces model deployment. Once we have our trained models ready for putting them into production, we will learn how to deploy them in different ways.

In module 5 we learn about model monitoring. Once the model is deployed, we need to make sure it is still performing well, and generate alerts or retrain the model if necessary.

Finally, module 6 covers best practices in MLOps to keep our code in a good condition.



## 5. MLOps maturity model

[Video source](https://www.youtube.com/watch?v=XwTH8BDGzYk).

The MLOps maturity model discussed in the video is based in this [Machine Learning operations maturity model article from Azure](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

The maturity model shows the continuous improvement in the creation and operation of a production level machine learning application environment. It helps clarify the Development Operations (DevOps) principles and practices necessary to run a successful MLOps environment.

The MLOps maturity model encompasses five levels of technical capability:

0. **No MLOps**
    * Notebooks, no proper pipelining, no proper experiment tracking, no metadata attached to model, no automation, ...
    * It can be useful for Proof of Concept (PoF) projects, where we don't need any automation.

1. **DevOps but not MLOps**
    * There is some level of automation related to best practices in software engineering: releases are automated, unit and integration tests, CI/CD, operational metrics.
    * But non of the operations are ML aware: there is no experiment tracking, no reproducibility, data scientists and data engineers work as separated teams, ...

2. **Automated model training**
    * Training pipeline, experiment tracking, some sort of model registry, still manual but low friction deployment.
    * DS and DE work together as a team.

3. **Automated model deloyment**
    * Very easy to deploy a model. It can be part of our data pipeline.
        ```
        Data preparation --> Model training --> Model deployment
        ```
    * We have A/B tests capabilites, so we can run and compare different versions of a model.
    * Model monitoring.

4. **Full MLOps autmation**
    * Highest level of automation.
    * Automated training, automated retraining, automated deployment.


Be aware that not all of our services, not all our models need to be at level 4 of MLOps maturity model. We should be pragamatic when deciding about it.