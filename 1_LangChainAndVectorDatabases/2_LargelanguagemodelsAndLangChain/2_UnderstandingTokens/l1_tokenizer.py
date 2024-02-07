'''
    Tokenization is a fundamental principle in natural language processing (NLP) that plays a crucial role in enabling language models to comprehend written information. 
    It entails breaking down textual inputs into individual units called tokens, forming the foundation for effectively understanding and processing language by neural networks. 

    The context length of the model is a frequently discussed characteristic among language models. 
    As an example, the GPT-3.5 model has a context length of 4096 tokens, covering both the tokens in the prompt and the subsequent completion. 
    Due to this constraint, it is advisable to be mindful of token usage when making requests to language models. 
    Nonetheless, various approaches exist to tackle this challenge when dealing with long-form inputs or multiple documents. 
    These methods include breaking up a lengthy prompt into smaller segments and sending requests sequentially, 
    or alternatively, submitting independent requests for each document and merging the responses in a final step. 

    The tokenization process involves creating a systematic pipeline for transforming words into tokens. Ir can be done on various levels:
        Character Level: Consider each character in a text as a token.
        Word Level: Encoding each word in the corpus as one token.
        Subword Level: Breaking down a word into smaller chunks when possible. 
            For example, we can encode the word “basketball” to the combination of two tokens as “basket” + “ball”.

    Subword-level encoding offers increased flexibility and reduces the number of required unique tokens to represent a corpus. 
    This approach enables the combination of different tokens to represent new words, eliminating the need to add every new word to the dictionary. 
    This technique proved to be the most effective encoding when training neural networks and LLMs. Well-known models like GPT family, LLaMA employ this tokenization method. 
    Therefore, our sole focus will be on one of its specific variants, known as Byte Pair Encoding (BPE). 

    Byte Pair Encoding (BPE)
        It is an iterative process to extract the most repetitive words or subwords in a corpus. 
        The algorithm starts by counting the occurrence of each character and builds on top of it by merging the characters. 
        It is a greedy process that carefully considers all possible combinations to identify the optimal set of words/subwords 
        that covers the dataset with the least number of required tokens.

        The next step involves creating the vocabulary for our model, which consists of a comprehensive dictionary comprising the most frequently occurring tokens 
        extracted by BPE (or another technique of your choosing) from the dataset. 
        The definition of a dictionary (dict type) is a data structure that holds a key and value pair for each row. 

        Due to the fact that neural networks only accept numerical inputs, we can utilize the vocabulary to establish a mapping between tokens and their corresponding IDs, 
        ike a lookup table. We have to save the vocabulary for future use cases to be able to decode the model's output from the IDs to words. 
        This is known as a pre-trained vocabulary, an essential component accompanying published pre-trained models. 
        Without the vocabulary, understanding the model's output (the IDs) would be impossible. 
        For smaller models like BERT, the dictionary can consist of as few as 30K tokens, while larger models like GPT-3 can expand to encompass up to 50K tokens.

    Tokens and Cost
        There is a direct correlation between the tokens and the cost in most proprietary APIs like OpenAI. 
        It is crucial to highlight that the prices will fall into two categories: the number of tokens in the prompt and the completion (the model's generation), 
        with the completion typically incurring higher costs. 
        For example, at the time of writing this lesson, GPT-4 will cost $0.03 per 1K tokens for processing your inputs, and $0.06 per 1K tokens for generation process. 
        As we saw in the previous lesson, you can use the get_openai_callback method from LangChain to get the exact cost, 
        or do the tokenization process locally and keep track of number of tokens as we see in the next section. For a rough estimation, as a general rule, 
        OpenAI regards 1K tokens to be approximately equal to 750 words.
'''

# In this section, we provide the codes to load the pre-trained tokenizer for the GPT-2 model from the Huggingface Hub using the transformers package. 

from transformers import AutoTokenizer

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# The code snippet above will grab the tokenizer and load the dictionary, so you can simply use the tokenizer variable to encode/decode your text. 
print( tokenizer.vocab )

# The next code sample will use the tokenizer object to convert a sentence into tokens and IDs.
token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")

print( "Tokens:   ", tokenizer.convert_ids_to_tokens( token_ids ) )
print( "Token IDs:", token_ids )

# Tokenizers Shortcomings
# Several issues with the present tokenization methods are worth mentioning.

# Uppercase/Lowercase Words: The tokenizer will treat the the same word differently based on cases. 
# For example, a word like “hello” will result in token id 31373, while the word “HELLO” will be represented by three tokens as [13909, 3069, 46] 
# which translates to [“HE”, “LL”, “O”].
# Dealing with Numbers: You might have heard that transformers are not naturally proficient in handling mathematical tasks. 
# One reason for this is the tokenizer's inconsistency in representing each number, leading to unpredictable variations. 
# For instance, the number 200 might be represented as one token, while the number 201 will be represented as two tokens like [20, 1].
# Trailing whitespace: The tokenizer will identify some tokens with trailing whitespace. 
# For example a word like “last” could be represented as “ last” as one tokens instead of [" ", "last"]. 
# This will impact the probability of predicting the next word if you finish your prompt with a whitespace or not. 
# As evident from the sample output above, you may observe that certain tokens begin with a special character (Ġ) representing whitespace, while others lack this feature.
# Model-specific: Even though most language models are using BPE method for tokenization, they still train a new tokenizer for their own models. 
# GPT-4, LLaMA, OpenAssistant, and similar models all develop their separate tokenizers.
