import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

import matplotlib.pyplot as plt

import torchvision

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


resnet = torchvision.models.resnet18()


device = "cuda"


class Retina:
    """A visual retina.

    Extracts a foveated glimpse `phi` around location `l`
    from an image `x`.

    Concretely, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        g: size of the first square patch.
        k: number of patches to extract in the glimpse.
        s: scaling factor that controls the size of
            successive patches.

    Returns:
        phi: a 5D tensor of shape (B, k, g, g, C). The
            foveated glimpse of the image.
    """

    def __init__(self, g, k, s):
        
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        
        phi = []
        size = self.g
        
        B, C, H, W = x.shape
        
        # New size is ratio-ed to image size: minimum of H, W, then divided by 2 * the number of glimpses
        size = int(min(H, W) / 5)
        og_size = int(min(H, W) / 5)
        
        # extract k patches of decreasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)
#             print("NEW SIZE: ", size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            phi[i] = torch.nn.functional.interpolate(phi[i], size = (og_size, og_size), mode = 'nearest')

        # concatenate into a single tensor, send to minConv, and flatten
        phi = torch.cat(phi, 0)

        return phi

    def extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.

        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2).
        size: a scalar defining the size of the extracted patch.

        Returns:
            patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape
                
        start = self.denormalize((W,H), l)
                        
        end = start + size

        # pad with zeros
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))

        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):
            
            cur_patch = x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]]
            
            if self.exceeds(from_x = start[i, 1], to_x = end[i, 1], from_y = start[i, 0], to_y = end[i, 0], H = H, W = W):
                new_coords = self.fix(from_x = start[i, 1], to_x = end[i, 1], from_y = start[i, 0], to_y = end[i, 0], H = H, W = W, size = size)
                cur_patch = x[i, :, new_coords[0] : new_coords[1], new_coords[2] : new_coords[3]]
            
            patch.append(cur_patch)

        return torch.stack(patch)

    
    def denormalize(self, dims, coords):
        
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
    
        W, H = dims
        x, y = coords[0]

        W = int(0.5 * (x + 1) * W)
        H = int(0.5 * (y + 1) * H)
                
        return torch.tensor([[W, H]], dtype = torch.long)

    
    def exceeds(self, from_x, to_x, from_y, to_y, H, W):
        """Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (from_x < 0) or (from_y < 0) or (to_x > H) or (to_y > W):
            return True
        return False

    
    def fix(self, from_x, to_x, from_y, to_y, H, W, size):
        
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        If it will exceed, make a list of the offending reasons and fix them
        """
        
        offenders = []
        
        if (from_x < 0):
            offenders.append("negative x")
        if from_y < 0:
            offenders.append("negative y")
        if from_x > H:
            offenders.append("from_x exceeds h")            
        if to_x > H:
            offenders.append("to_x exceeds h")
        if from_y > W:
            offenders.append("from_y exceeds w")
        if to_y > W:
            offenders.append("to_y exceeds w")            
            
        
        if ("from_y exceeds w" in offenders) or ("to_y exceeds w" in offenders):
            from_y, to_y = W - size, W
            
        if ("from_x exceeds h" in offenders) or ("to_x exceeds h" in offenders):
            from_x, to_x = H - size, H     
            
        elif ("negative x" in offenders):
            from_x, to_x = 0, 0 + size
            
        elif ("negative y" in offenders):
            from_y, to_y = 0, 0 + size            

        return from_x, to_x, from_y, to_y


class GlimpseNetwork(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()

        self.retina = Retina(g, k, s)

        # Glimpse layer (dims are [k, final_layer_size, 1, 1])
        self.fc1 = nn.Linear(k * h_g, h_g)

        # location layer (D_in is the number of zoomed in glimpses)
        self.fc2 = nn.Linear(k, h_l)
        
        # Final fully connected layers for what & where
        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)
        
                
        # Resnet18 convolutional layers for the glimpse 
        self.conv1_miniConv = resnet.conv1.to(device)
        self.bn1_miniConv = resnet.bn1.to(device)
        self.relu_miniConv = resnet.relu.to(device)
        self.maxpool_miniConv = resnet.maxpool.to(device)
        self.layer1_miniConv = resnet.layer1.to(device)
        self.layer2_miniConv = resnet.layer2.to(device)
        self.adp_pool_miniConv = torch.nn.AdaptiveAvgPool2d((1, 1)).to(device)        
                
        
    def forward(self, x, l_t_prev):
                
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        
        # minConv layers (these need to be here in order to be registered as trainable model paramerters)
        phi = self.conv1_miniConv(phi)
        phi = self.bn1_miniConv(phi)
        phi = self.relu_miniConv(phi)
        phi = self.maxpool_miniConv(phi)
        phi = self.layer1_miniConv(phi)
        phi = self.layer2_miniConv(phi)
        phi = self.adp_pool_miniConv(phi)
        phi = phi.flatten(start_dim = 0)
        phi_out = F.relu(self.fc1(phi)) # feed phi to respective fc layer

        # flatten location vector & feed to respective fc layer
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)        
        l_out = F.relu(self.fc2(l_t_prev))

        # Final fully connected layers 
        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # Add together what & where and activate
        g_t = F.relu(what + where)

        return g_t


class LocationNetwork(nn.Module):
    """
    
    The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
            
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(input_size = input_size,
                                   hidden_size = hidden_size,
                                   num_layers = 2,
                                   bias = True,
                                   batch_first = True)
        
    def forward(self, gn, rln_hs_prev, rln_cs_prev, gn_prev): 
                
        # If it's the first timestep, no previous glimpse information to send to LSTM 
        if gn_prev == None:
            gn = gn.unsqueeze(0) # add a dimension for the glimpse timestep
            rln, (rln_hs_cur, rln_cs_cur) = self.lstm1(gn)
        else:
            
#             print("gn_prev input shape: ", gn_prev.shape)  
            
            gn = torch.cat((gn, gn_prev), dim = 0).unsqueeze(0)   
#             print("gn.shape", gn.shape)
            rln, (rln_hs_cur, rln_cs_cur) = self.lstm1(gn, (rln_hs_prev, rln_cs_prev))
                
        return rln, rln_hs_cur, rln_cs_cur


class ClassifierNetwork(nn.Module):
    """

    The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: output probability vector over the classes.
        
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc_class1 = nn.Linear(input_size, output_size)
        self.fc_cont = nn.Linear(output_size, 1)
        
        
    def forward(self, rln_hs):
        
        # Take the lower recurrent layer
        rnn_layer0 = rln_hs[0, :, :]
               
        # Input a_t into classifer for preds for each class for num_glimpses == 2, shape will be [2, 256]
        a_t_class = F.log_softmax(self.fc_class1(rnn_layer0), dim = 1)#.flatten(start_dim = 0) # shape goes from [2,2] -> [1,4]
        a_t_cont = self.fc_cont(a_t_class)

        return a_t_class, a_t_cont


class EmissionNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std
                
        # Flattened specs for 2 glimpses
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)        

    def forward(self, rln_hs_cur):
        
        # Grab the hidden layer state from the second layer of the LSTM
        last_hl = rln_hs_cur[1, :, :]
                                                                
        # compute mean
        feat = F.relu(self.fc(last_hl.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        # reparametrization trick
        new_loc = torch.distributions.Normal(mu, self.std).rsample()
        new_loc = new_loc.detach()
        log_pi = Normal(mu, self.std).log_prob(new_loc)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        new_loc = torch.clamp(new_loc, -1, 1)
        
        return log_pi, new_loc


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        
        input_size = 1 * 256
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, rln_hs_cur):
        rln_hs_cur = rln_hs_cur[1, :, :]
        b_t = self.fc(rln_hs_cur.detach()).unsqueeze(0)
        return b_t
