There are two stages, search and evaluation.
There are six models. The search stage code of each model is in the directory with the model name.
For example, "darts" includes the search stage code of the DARTS model, "darts-lfm" includes the search stage code of the LFM DARTS model.
Search code for models "darts", "pcdarts", "pdarts" are adaptded from the original code published with the DARTS, P-DARTS, PC-DARTS papers
to work with PyTorch 1.8.0.
Search code for models "darts-lfm", "pcdarts-lfm", "pdarts-lfm" include implementations for the LFM model in "search.py".

Evaluate code is combined from 3 sets of evaluate code in "darts", "pcdarts", "pdarts" to avoid repetitive code.

Use commands in "search.sh" and "train.sh" in "shyaml" directories to and search and train, respectively. "shyaml" stands for shell scripts and yaml files.
