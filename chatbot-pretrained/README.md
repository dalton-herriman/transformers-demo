# GPT-2 Chatbot Demo

This project provides a simple command-line chatbot using a pretrained GPT-2 model from Hugging Face Transformers. It supports optional finetuning on custom datasets and interactive chatting.

## Features

- Loads GPT-2 and tokenizer with GPU support if available
- Interactive chat session with context-aware responses
- Optional finetuning on Hugging Face datasets or custom prompt/response pairs
- Configurable generation parameters (sampling, temperature, top-k, top-p)

## Requirements

- Python 3.7+
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [torch](https://pytorch.org/)

Install dependencies:
```bash
pip install torch transformers datasets
```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

You will be prompted to optionally finetune the model. You can specify a Hugging Face dataset or use the default small set.

Type your messages at the prompt. Type `exit` to quit.

## Example

```
Simple GPT-2 Chatbot. Type 'exit' to quit.
Do you want to finetune the model? (y/n): n
You: Hello!
Bot: Hi there!
You: How are you?
Bot: I'm a chatbot, but I'm doing well!
You: exit
```

## Customization

- To use a different GPT-2 variant, change the `model_name` in `load_model_and_tokenizer()`.
- For custom finetuning data, modify the `train_data` list or provide your own dataset.

## Technical Considerations

 - I wasn't originally going to finetune this model but the output from the default GPT-2 pretrained was horrendously unusable so it became necessary.

## License

This project is for educational/demo purposes. Refer to the respective licenses for GPT-2 and datasets used.
