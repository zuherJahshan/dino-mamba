# Will define the following funcitonalities
from einops import rearrange, repeat
import torch

class Codebook(object):
    def __init__(
        self,
        dim: int,
        num_codewords: int,
        layers: int,
        device: str = "cuda",
        decay: float = 0.9,
    ):
        # set props
        self.dim = dim
        self.num_codewords = num_codewords
        self.decay = decay
        self.device = device

        # generate the codebook from a random normal distribution
        self.codebooks = []
        self.codebooks_sum = []
        self.codebooks_cnt = []
        codebook_var = 1 / (self.num_codewords ** 0.5)
        for _ in range(layers):
            self.codebooks.append(torch.normal(0.0, codebook_var, [num_codewords, dim]).to(device))
            self.codebooks_sum.append(self.codebooks[-1].clone())
            self.codebooks_cnt.append(torch.ones(num_codewords).to(device))


    def get_decay(self):
        return self.decay
    

    def set_decay(
        self,
        decay: float,
    ) -> None:
        self.decay = decay


    def get_closest_codewords(
        self,
        x: torch.Tensor, # gets a tensor of size [B,d]
        codebook_idx: int,
    ) -> torch.Tensor: # Returns a tensor of size [B] where each element is the index of the closest codeword (i.e., in {0,1,...,N-1})
        # codebook is of shape [N,d], rearrange to be [1,N,d]
        # x is of shape [B,d], rearrange to be [B,1,d]
        re_x = rearrange(x, 'B d -> B 1 d')
        re_codebook = rearrange(self.codebooks[codebook_idx], 'N d -> 1 N d')
        
        # compute the distance between the two tensors, and loss any ra dimensions
        dist = torch.cdist(re_x, re_codebook) # will result an input of shape BxNx1
        dist.squeeze()

        # return the index of the closest codeword
        return torch.argmin(dist, dim=-1).squeeze()
        

    def update_codewords(
        self,
        x: torch.Tensor, # gets a tensor of size [B,d]
        closest_codewords: torch.Tensor, # gets a tensor of size [B]
        codebook_idx: int,
    ) -> None:
        # Make closest codewords a one-hot tensor
        one_hot = torch.nn.functional.one_hot(closest_codewords, self.num_codewords)
        one_hot = one_hot.squeeze().float() # tensor of shape B,N
        
        # Calculate the new codebooks sum
        z = torch.einsum('bd,bn->nd', x, one_hot)
        cnts = torch.sum(one_hot, dim=0) # tensor of shape N, representing the neighbors of each codeword

        tau = torch.where(cnts > 0, self.decay, 1).to(self.device) ## freeze the codeword if it has no members

        # Update the codebook sum
        self.codebooks_sum[codebook_idx] = tau * self.codebooks_sum[codebook_idx] + (1 - tau) * z
        self.codebooks_cnt[codebook_idx] = tau * self.codebooks_cnt[codebook_idx] + (1 - tau) * cnts

        # Update the codebook
        self.codebooks[codebook_idx] = self.codebooks_sum[codebook_idx] / self.codebooks_cnt[codebook_idx].unsqueeze(1)

        return self.codebooks[codebook_idx]
        
    def save_state(self):
        return {
            'codebooks': [cb.cpu().numpy() for cb in self.codebooks],
            'codebooks_sum': [cbs.cpu().numpy() for cbs in self.codebooks_sum],
            'codebooks_cnt': [cbc.cpu().numpy() for cbc in self.codebooks_cnt],
        }

    def load_state(self, state):
        self.codebooks = [torch.tensor(cb).to(self.device) for cb in state['codebooks']]
        self.codebooks_sum = [torch.tensor(cbs).to(self.device) for cbs in state['codebooks_sum']]
        self.codebooks_cnt = [torch.tensor(cbc).to(self.device) for cbc in state['codebooks_cnt']]
