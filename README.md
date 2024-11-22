# TinyGPT
A PyTorch re-implementation of GPT (GPT-2 compatible), both training and inference. For educational purpose only.

#### Training
the training of the model is hard-coded to use the "data/PG1_text.txt" to demo the training process
```
python trainer.py
```

#### Inference
the input prompt is hard-coded in tinygpt.py
```
python tinygpt.py
```

#### Next Step
- Load GPT-2 weights
- Train with Gutenberg corpus
- Fine-tuning for different task(s)
- Add distributed training support
