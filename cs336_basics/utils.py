from jaxtyping import Float
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_values = in_features.amax(dim=dim, keepdim=True)
    adjusted_values = (in_features - max_values).exp()
    sums = adjusted_values.sum(dim=dim, keepdim=True)

    return adjusted_values / sums
