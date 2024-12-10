import sys
import random
import numpy as np
import torch


def set_seed(seed):
    """
        set global seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def logging(path, message, print_=True):
    if print_:
        print(message, file=sys.stderr)
    if path:
        with open(path, 'a+') as f:
            f.write(message + '\n')


def dump(lines, path, stdout=True):
    if stdout:
        print(*lines, sep='\n', file=sys.stdout)
    elif path:
        with open(path, 'a+') as f:
            f.writelines(line + '\n' for line in lines)


def write_z(z, path):
    with open(path, 'w') as f:
        for zi in z:
            for zij in zi:
                f.write('%f ' % zij)
            f.write('\n')


def lerp(t, p, q):
    return (1-t) * p + t * q

# spherical interpolation https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
def slerp(t, p, q):
    o = np.arccos(np.dot(p/np.linalg.norm(p), q/np.linalg.norm(q)))
    so = np.sin(o)
    return np.sin((1-t)*o) / so * p + np.sin(t*o) / so * q

def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0*i/(n-1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)
