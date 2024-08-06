from typing import List
import torch
import markdown

class OutConvShape:
    def __init__(
        self,
        convolutional_layers,
        same: bool = True,
    ):
        self.alpha = 1
        self.beta = 0
        for i in range(len(convolutional_layers)):
            self.alpha *= convolutional_layers[i]['s'] # Assuming stride is the second element
            tmp_beta = convolutional_layers[i]['k']
            for j in range(i+1, len(convolutional_layers)):
                tmp_beta *= convolutional_layers[j]['s']
            self.beta += tmp_beta
        if same:
            self.beta = 0

    def __call__(self, x):
        return torch.ceil((x - self.beta) / self.alpha).to(torch.int)
    
    @staticmethod
    def print_description():
        markdown_string = """
if you have a series of n convolution layers, each convolution, i, has k_i, s_i, as a kernel and a stride. The output of the ith layer is given by:
$$ x_{i+1} = \frac{x_i - k_i}{s_i} $$
Also, itcan be re-written as:
$$ s_i x_{i+1} = x_i - k_i $$
where
$$ s_{i-1} x_i = x_{i-1} - k_{i-1} $$
This implies that:
$$ s_i s_{i-1} x_{i+1} = x_{i-1} - s_{i-1}k_i - k_{i-1}$$

And from here we can derive a general formula for the output of the nth layer, as a function of the input of first layer:
$$
\Pi_{i=1}^{n-1}{s_i} x_n = x_1 - \sum_{i=1}^{n-1}{\Pi_{j=i+1}^{n-1}{s_j}k_i}
$$

Note that the Sigma-Pi is not input dependent and hence, it is a constant depending only on the network architecture. This is an important quality, since it makes the computation of the relevant output window an O(1) operation.

To deal with boundries, we can write the following formula:
$$
\alpha = \Pi_{i=1}^{n-1}{s_i} \\
\beta = \sum_{i=1}^{n-1}{\Pi_{j=i+1}^{n-1}{s_j}k_i} \\
x_n \ge \lceil \frac{x_1 - \beta}{\alpha} \rceil
$$
where $\alpha$ and $\beta$ are constants depending only on the network architecture. Noting that x_n is almost equal to $\lceil \frac{x_1 - \beta}{\alpha} \rceil$ 
"""
        print(markdown.markdown(markdown_string))

def get_starting_mask_prob(mask_prob, mask_length):
    # P(not masked) = # all previous max_len elements are not a starting mask
    # P(not masked) = (1 - p(starting_mask)) ^ (mask_length)
    # P(masked) = 1 - P(not masked)
    # P(masked) = 1 - (1 - p(starting_mask)) ^ (mask_length)
    # p(starting_mask) = 1 - (1 - P(masked)) ^ (1/(mask_length ))
    return 1 - (1 - mask_prob) ** (1/(mask_length))