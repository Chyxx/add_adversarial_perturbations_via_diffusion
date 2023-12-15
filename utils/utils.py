import torch
from improved_diffusion import dist_util


def one_hot(x, class_count):
    return torch.eye(class_count, device=dist_util.dev())[x, :]


def norm(x):
    """Limit value between 0 and 1"""
    return torch.clamp(x, 0, 1)


def hinge_func(x, class_count, t):
    """目标攻击时的损失函数"""
    k = 0
    # label = one_hot(torch.full([x.size(0)], t, dtype=torch.int), class_count).type(torch.int)
    # tmp = torch.where(label == 1, float('-inf'), x)
    # tmp = torch.log(tmp.max(dim=1)[0]) - torch.log(x[:, t])
    # return torch.where(tmp > -k, tmp, -k)
    idx = list(range(class_count))
    idx.pop(t)
    idx = torch.tensor(idx).cuda()
    tmp = torch.log(x.index_select(1, idx).max(dim=1)[0]) - torch.log(x[:, t])
    return torch.clamp(tmp, min=-k)

def untargeted_hinge_func(x, class_count, labels):
    """非目标攻击时的损失函数"""
    k = 5
    # label = one_hot(torch.full([x.size(0)], t, dtype=torch.int), class_count).type(torch.int)
    # tmp = torch.where(label == 1, float('-inf'), x)
    # tmp = torch.log(tmp.max(dim=1)[0]) - torch.log(x[:, t])
    # return torch.where(tmp > -k, tmp, -k)
    target = one_hot(labels, class_count)
    x = torch.log(x)
    with torch.no_grad():
        t = x[target == 1].unsqueeze(dim=1)
    t = t - x + k
    h = torch.where(torch.logical_or(target == 1, t <= 0), 0, t)
    return h.sum(dim=1)
    # tmp = torch.log(tmp) - torch.log(torch.where(x == tmp, 0, x).max(dim=1)[0])
    # return torch.clamp(tmp, min=-k)


def get_prob(x):
    return torch.chunk(x, dim=1, chunks=2)[1]
