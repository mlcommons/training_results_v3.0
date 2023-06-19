import torch
from torch.autograd import Function

def focalloss(input, target, reduction):

    if torch.is_grad_enabled():
        return FocalLossFunc.apply(input, target, reduction)
    return torch.ops.torch_ipex.focal_loss_forward(input, target, reduction)

class FocalLossFunc(Function):
    @staticmethod
    def forward(ctx, input, target, reduction):
        if input.requires_grad:
            ctx.save_for_backward(input, target)
            ctx.reduction=reduction
        output = torch.ops.torch_ipex.focal_loss_forward(input, target, reduction)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, target= ctx.saved_tensors
        grad_in = torch.ops.torch_ipex.focal_loss_backward(grad_out.contiguous(), input, target, ctx.reduction)
        return grad_in,None,None
