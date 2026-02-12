# Data
- data setup, i.e. FeatureStore and encoding logic, is currently reliant on python indexing and loops which will be highly inefficient (single-threaded, loop overheads in python, memory limitations). Refactor this significantly, probably in rust / C, in a way that will be faster and ideally work in cases where user / item table is larger than RAM. 
- Add data prep script for h & m.

# Modelling
- Ability to use text and image encoders for cols given in config / argparser, with use of an embedding model from HuggingFace. This should contain an option to either fine-tune the embeddings or freeze. 
- Use torchrec style embeddings for better handling at scale.
- Add option for mixed precision training. Should be an option in config / argparser to be applied during training.
- Add option for early stopping, defined in config / argparser. 
- Optional selection of what optimizer to use in the config / argparser. 
- Add mixed precision training.
- LRFinder that will calculate optimal LR for a given model / dataset.
- Scaling laws to define correct embedding dims.
- Sequential column processing. A few ways to do this - either has sequences as rows (inefficient in terms of data storage but simpler) 

# Deployment 
- Test actual live API deployment on a server. Need to work out how that would work.
- App for viewing predictions (probably gradio). 

# Documentation
- The metrics here use splitting based on time T. I think I like this approach the most as it replicates production environments more closely (i.e. in reality you're predicting what will happen in a specific future time period from now, rather than an arbitrary distance in the past). However Leave-One-Out is more common in academic research. I've also found that how to calculate metrics seems to massively differ in approaches, e.g. recall@k can either be pct of k hits that are relevant or 1/0 if any predicted are relevant. I don't fully get this yet so I want to understand better first, but will document properly when I do (or fix metrics to better reflect industry standards)   