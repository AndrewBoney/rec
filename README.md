# rec

Recommendation systems are hard. They require complex dataset construction from diverse sources, multiple training schedules, frequent mismatches between training objectives and deployment KPIs, custom model architectures, and versioning across multiple dimensions (data, code, and model weights), among other challenges.

Because of this complexity, it’s surprisingly difficult to find simple, end-to-end implementations that researchers can follow. Instead, many practitioners end up stitching together systems from tutorials that only cover isolated parts of the recommendation workflow. This often leads to fragmented understanding and brittle implementations.

This repository is designed to help address that gap by providing a **complete, end-to-end recommendation system**, covering data preparation, training, and deployment in a single, coherent example.

The project is inspired by Andrej Karpathy’s [nanochat](https://github.com/karpathy/nanochat), which aims to be the minimal viable implementation of a modern LLM. In the same spirit, this repo provides a **lightweight recommendation system** that does only what is necessary—while remaining simple enough for researchers to understand, modify, and extend.

__What this *is*__

- A learning resource for researchers who want to understand the full recommendation system workflow, end to end.
- A collection of clean, reusable implementations of the trickier parts of building recommendation systems.
- A foundation for experimenting with, and contributing, new or interesting ideas in the recommender systems space.

__What this *isn’t*__

- An exhaustive survey or implementation of all recommendation approaches. The focus here is deliberately narrow, primarily on **dual-encoder ('two-tower') architectures**.
- A fully production-ready system. While the code aims to be well-written, tested, and reusable, the primary goal is **clarity and minimalism**. The target is to keep the project under ~5,000 lines of readable Python. A real-world production system will be orders of magnitude larger and span multiple languages and services.