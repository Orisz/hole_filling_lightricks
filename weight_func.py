from numpy import linalg


class Weight:
    def __init__(self, z, eps):
        self.z = z
        self.eps = eps

    def get_weight(self, pix_u, pix_v):
        l1_norm = linalg.norm((pix_u - pix_v), ord=1)
        pow_norm = l1_norm ** self.z
        w = 1.0 / (pow_norm + self.eps)
        return w
