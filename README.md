# rec

Recommendation Systems are very very difficult. They require complex dataset construction from diverse sources, multiple training schedules, mismatches between training objectives and KPI's at deployment, custom model architectures, versioning across multiple dimensions (data, model code, model weights), etc. etc. 

As such, it's very difficult to find examples of simple end to end implementations that researchers can follow. This leads to researchers cobbling together systems from tutorials and examples that only cover certain aspects of the Rec-Sys workflow, leading to muddled understanding and implementation. 

This repo is designed to help solve that problem, by implementing an end to end recommendation system - including data prep, training and deployment. This is inspired by Andrej Karpathy's [nanochat project](https://github.com/karpathy/nanochat), which is a minimal as possible implementation of a modern LLM end to end. In a similar way I plan to provide a lightweight RecSys, that will do just about what is neccesary, while being simple enough that researchers can easily adjust and intervene. 

What this:

- A piece for researchers to better understand the full Recommendation System workflow.
- Good reusable code implementating of the tricker aspects of Recommendation Systems.
- An opportunity for researchers to contribute implementations of new or interesting ideas in the rec space. 

What this isn't:

- An exaughstive implementation of all approaches to recs. My focus is specifically on Duel Encoder networks.  
- A full production ready implementation. While I aim to ensure code is well written, tested and reusable, a key goal here is to be minimal in approach. I'd like to keep this < 5000 lines of readable python code, where a real production implementation at scale in industry may well be 10x that across multiple languages.