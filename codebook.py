# Will define the following funcitonalities
from einops import rearrange, repeat, einsum
import torch
import math

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
        self.layers = layers

        # generate the codebook from a random normal distribution
        self.codebooks = []
        self.codebooks_sum = []
        self.codebooks_cnt = []
        codebook_var = 1 / math.sqrt(self.dim)

        
        self._reset_accumulated()


        for _ in range(layers):
            cb = torch.normal(0.0, codebook_var, [num_codewords, dim]).to(device)
            # cb = torch.nn.functional.normalize(cb, p=2, dim=-1)
            self.codebooks.append(cb)
            self.codebooks_sum.append(self.codebooks[-1].clone())
            self.codebooks_cnt.append(torch.ones(num_codewords).to(device))

    def _reset_accumulated(self):
        self.accumulated_closest_codewords = [None] * self.layers
        self.accumulated_x = [None] * self.layers
        self.mask = [None] * self.layers


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
        

    def accumulate_codewords_for_update(self, codebook_idx, x, closest_codewords, mask):
        if self.accumulated_closest_codewords[codebook_idx] is None:
            self.accumulated_closest_codewords[codebook_idx] = closest_codewords
            self.accumulated_x[codebook_idx] = x
            self.mask[codebook_idx] = mask
        else:
            self.accumulated_closest_codewords[codebook_idx] = torch.cat([self.accumulated_closest_codewords[codebook_idx], closest_codewords], dim=0)
            self.accumulated_x[codebook_idx] = torch.cat([self.accumulated_x[codebook_idx], x], dim=0)
            self.mask[codebook_idx] = torch.cat([self.mask[codebook_idx], mask], dim=0)


    def update_codewords(self) -> None:
        # Make closest codewords a one-hot tensor
        for codebook_idx, (acc, ax) in enumerate(zip(self.accumulated_closest_codewords, self.accumulated_x)):
            if acc is None or ax is None:
                raise ValueError("You should call accumulate_codewords_for_update before calling update_codewords")
            one_hot = torch.nn.functional.one_hot(acc, self.num_codewords)
            one_hot = one_hot.squeeze().float() # tensor of shape B,N
            mask = self.mask[codebook_idx].float() # tensor of shape B
            
            # Calculate the new codebooks sum
            z = einsum(one_hot, mask, 'b n, b -> b n')
            z = einsum(ax, z, 'b d, b n -> n d')


            cnts = torch.sum(one_hot, dim=0) # tensor of shape N, representing the neighbors of each codeword

            tau = torch.where(cnts > 0, self.decay, 1).to(self.device) ## freeze the codeword if it has no members

            # Update the codebook sum
            self.codebooks_sum[codebook_idx] = tau.unsqueeze(-1) * self.codebooks_sum[codebook_idx] + (1 - tau.unsqueeze(-1)) * z
            self.codebooks_cnt[codebook_idx] = tau * self.codebooks_cnt[codebook_idx] + (1 - tau) * cnts

            # Update the codebook
            self.codebooks[codebook_idx] = self.codebooks_sum[codebook_idx] / self.codebooks_cnt[codebook_idx].unsqueeze(1)

        self._reset_accumulated()
        

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
        # eza r7 n3mlha blknise msh r7 tetwza3 zy elmaskene elka3ke