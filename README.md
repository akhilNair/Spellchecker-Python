# Spellchecker-Python
RNN for spelling correction
1. 2 parts in RNN model
  a. The first part is an encoding part that contains a list of cells. Each cell takes two input, one input is the current character and        the other one is the output from the previous cell.
  b. The second part is a decoding part that also contains a list of cells. Each cell takes one input from the previous cell and generates      a vector as the output. The vector corresponding to a character in the vocabulary.
 
 Details 
 1. Every input is one character since I don’t want to deal with tokenizer.
