## Kraken Architecture

![alt text](https://github.com/cognitivecomputations/kraken/blob/main/kraken.png?raw=true)

## Overview

The Kraken Architecture is a sophisticated machine learning framework designed for dynamic text generation tasks. It utilizes the Hugging Face transformers library to orchestrate multiple causal language models (CLMs) and intelligently route input through different models based on the context and content of the input text. The architecture is powered by a custom configuration class (KrakenConfig) that facilitates the integration and management of various components such as tokenizers, models, and routing mechanisms.

## Features

Dynamic Model Routing: Uses a sequence classification model to route inputs to the most suitable language model based on the input's characteristics.
Multiple Language Models: Supports integration of various pre-trained causal language models, allowing for flexible, context-appropriate responses.
Customizable Templates: Includes support for input formatting using predefined templates, enhancing the model's adaptability to different conversational contexts.
Extensible Configuration: Leverages a custom configuration setup that can be easily extended and adapted for various use cases involving causal language modeling.

## Requirements

Python 3.11+
transformers 4.40+
torch 2.2+

## How to Use

(Optional) 0. Run the jupyter notebook kraken_train_router.ipynb to train a router that will be imported later as a our router on the Kraken CoE Architecture


1. Run the kraken_lm_save.ipynb that will load a router (could be the one you have trained in step 0.) and sets up a model following the Kraken CoE Architecture, according to the config.json.
This will generate a subfolder ./kraken_model


3. Run kraken_lm_load.ipynb to understand how to load the newly created model

## Cite As

Fernando Fernandes Neto, David Golchinfar, Lucas Atkins, Eric Hartford - Kraken: An OpenSource Collection of Experts Model, 2024
