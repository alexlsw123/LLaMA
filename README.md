# LLaMA2

### Important Notes
* Followed `setup.sh` to properly setup the environment and install dependencies.
* I only used libraries that are installed by `setup.sh`, no other external libraries were used (e.g., `transformers`).

## Pre-trained weight/data
The pre-trained weights for my language model were loaded from `stories42M.pt`; an 8-layer, 42M parameter language model pre-trained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (a dataset of machine-generated children's stories). This model is small enough that it can be trained (slowly) without a GPU. I used Colab (personal GPU machine also works) to be able to iterate more quickly.

## Task
By implementing the core component of the LLaMA2 model (`llama.py`, `classifier.py`, and `optimizer.py`), the task was to perform sentence classification on `sst` dataset and `cfimdb` dataset with the model. 

## Result

I tested the model in following 3 settings once I implemented the model:
1) Generated a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). I saw coherent, grammatical English being generated (though the content and topicality of the completion was somewhat absurd, since this LM was pretrained exclusively on children's stories).
2) Performed zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Performed task-specific finetuning of my Llama2 model, after implementing a classification head in `classifier.py`. This gave much stronger classification results.

### outputs/accuracies: 

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. I generated two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.498 (0.000)
Test Accuracy: -

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.
