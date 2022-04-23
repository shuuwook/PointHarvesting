import torch
from torch.autograd import Variable, grad
from utils.parallel import Reduce

class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, multi_gpu=False):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.multi_gpu = multi_gpu

    def __call__(self, netD, real_data, fake_data):
        ################################### Randomly mix real and fake data ##################################
        if self.multi_gpu is False:
            real_data = [real_data]
            fake_data = [fake_data]

        interpolates = []
        for i in range(len(real_data)):
            B = real_data[i].size(0)
            alpha = torch.rand(B, 1, 1, requires_grad=True).to(real_data[i].device)
            interpolates.append(real_data[i].data + alpha * (fake_data[i].data - real_data[i].data))
        ######################################################################################################

        ############################# Compute output of D for interpolated input #############################
        if self.multi_gpu:
            disc_interpolates = netD(interpolates)
        else:
            disc_interpolates = netD(*interpolates)
            disc_interpolates = [disc_interpolates]
        ######################################################################################################
        
        ########################## Compute gradients w.r.t the interpolated outputs ##########################
        gradient_penalty = []
        for i in range(len(interpolates)):
            B = interpolates[i].size(0)

            gradients = grad(outputs=disc_interpolates[i], inputs=interpolates[i],
                             grad_outputs=torch.ones(disc_interpolates[i].size()).to(interpolates[i].device),
                             create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(B,-1)
            gradient_penalty.append((((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2))
        
        gradient_penalty = (Reduce.apply(*gradient_penalty) / len(gradient_penalty)).mean() * self.lambdaGP
        #######################################################################################################
        
        return gradient_penalty