# Spellchecker-Python
RNN for spelling correction
1. 2 parts in RNN model
  a. The first part is an encoding part that contains a list of cells. Each cell takes two input, one input is the current character and        the other one is the output from the previous cell.
  b. The second part is a decoding part that also contains a list of cells. Each cell takes one input from the previous cell and generates      a vector as the output. The vector corresponding to a character in the vocabulary.
 
 Details 
 1. Every input is one character since I don’t want to deal with tokenizer.
 
 
 http://adventuresinmachinelearning.com/wp-content/uploads/2017/09/LSTM-diagram.png
 Notice first, on the left hand side, we have our new word/sequence value $x_t$ being concatenated to the previous output from the cell $h_{t-1}$. The first step for this combined input is for it to be squashed via a tanh layer. The second step is that this input is passed through an input gate. An input gate is a layer of sigmoid activated nodes whose output is multiplied by the squashed input. These input gate sigmoids can act to “kill off” any elements of the input vector that aren’t required. A sigmoid function outputs values between 0 and 1, so the weights connecting the input to these nodes can be trained to output values close to zero to “switch off” certain input values (or, conversely, outputs close to 1 to “pass through” other values).
The next step in the flow of data through this cell is the internal state / forget gate loop. LSTM cells have an internal state variable $s_t$. This variable, lagged one time step i.e. $s_{t-1}$ is added to the input data to create an effective layer of recurrence. This addition operation, instead of a multiplication operation, helps to reduce the risk of vanishing gradients. However, this recurrence loop is controlled by a forget gate – this works the same as the input gate, but instead helps the network learn which state variables should be “remembered” or “forgotten”.


 
Finally, we have an output layer tanh squashing function, the output of which is controlled by an output gate. This gate determines which values are actually allowed as an output from the cell $h_t$.

The mathematics of the LSTM cell looks like this:

Input

First, the input is squashed between -1 and 1 using a tanh activation function. This can be expressed by:

$$g = tanh(b^g + x_tU^g + h_{t-1}V^g)$$

Where $U^g$ and $V^g$ are the weights for the input and previous cell output, respectively, and $b^g$ is the input bias. Note that the exponents g are not a raised power, but rather signify that these are the input weights and bias values (as opposed to the input gate, forget gate, output gate etc.).

This squashed input is then multiplied element-wise by the output of the input gate, which, as discussed above, is a series of sigmoid activated nodes:

$$i = \sigma(b^i + x_tU^i + h_{t-1}V^i)$$

The output of the input section of the LSTM cell is then given by:

$$g \circ i$$

Where the $\circ$ operator expresses element-wise multiplication.

Forget gate and state loop

The forget gate output is expressed as:

$$f = \sigma(b^f + x_tU^f + h_{t-1}V^f)$$

The output of the element-wise product of the previous state and the forget gate is expressed as $s_{t-1} \circ f$. The output from the forget gate / state loop stage is:

$$s_t = s_{t-1} \circ f + g \circ i$$

Output gate

The output gate is expressed as:

$$o = \sigma(b^o + x_tU^o + h_{t-1}V^o)$$

So the final output of the cell , with the tanh squashing, can be shown as:

$$h_t = tanh(s_t) \circ o$$
