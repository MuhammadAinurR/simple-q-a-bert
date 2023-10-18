# Simple Question-Answering System with BERT
This repository contains a simple question-answering system built with the BERT model. The model is fine-tuned on the SQuAD dataset and can answer questions based on a given context.

# Features
* Uses the bert-large-uncased-whole-word-masking-finetuned-squad model from Hugging Face’s transformers library.
* The system takes a context (a piece of text) and a question, and returns an answer based on the context.
* The answer is extracted directly from the context.

# Usage
The main function in this repository is robot(context, question). Here’s how to use it:

from transformers import BertForQuestionAnswering, BertTokenizer

    import torch

# Load the pre-trained BERT model and tokenizer
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

# Context of the story
    story = """
    ... Your story here ...
    """

# Question to ask about the story
    question = "Your question here"

    def robot(context, question):

        # Tokenize the input text and question
        inputs = tokenizer(question, context, return_tensors="pt")

        # Get the answer
        output = model(**inputs)

        # Get start and end scores for answer
        answer_start_scores = output.start_logits
        answer_end_scores = output.end_logits

        # Find start and end of answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Convert tokens to string
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
        return answer

    answer = robot(story, question)
    print("Answer:", answer)

Replace "Your story here" and "Your question here" with your own context and question. The robot() function will return an answer based on the context.

Requirements
Python 3.6 or later.
PyTorch 1.0.0 or later.
Transformers library from Hugging Face.
Installation
You can install the required packages with pip:

    pip install torch transformers
