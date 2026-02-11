# Data
- Add data prep script for h & m.

# Modelling
- Ability to use text and image encoders for cols given in config / argparser, with use of an embedding model from HuggingFace. This should contain an option to either fine-tune the embeddings or freeze. 
- Use torchrec style embeddings for better handling at scale.
- Add option for mixed precision training. Should be an option in config / argparser to be applied during training.
- Add option for early stopping, defined in config / argparser. 
- Optional selection of what optimizer to use in the config / argparser. 
- LRFinder that will calculate optimal LR for a given model / dataset.
- Scaling laws to define correct embedding dims.
- Sequential column processing. A few ways to do this - either has sequences as rows (inefficient in terms of data storage but simpler) 

# Deployment 
- Test actual live API deployment on a server. Need to work out how that would work.
- App for viewing predictions (probably gradio). 