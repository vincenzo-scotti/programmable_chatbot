# Models

This directory is used to host the trained model(s).
These directories contain both weights checkpoints and tokenisers.
It is possible to download a zip archive with all the trained models and tokenisers at this [link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/Efr1JtJNCARPuLRZXzDz04MBzl-hghON_FFwahi-lxEZBA?e=K1JoNi).

All model have been trained fine-tuning [GPT-2](https://openai.com/blog/tags/gpt-2/) using the version released by [HuggingFace](https://huggingface.co) through their [Transformers](https://huggingface.co/docs/transformers/index) package. 
Refer to the chatbot API to use the models.

Directory structure:
```
 |- models/
   |- ppm_dlm/
     |- config.json
     |- pytorch_model.bin
```