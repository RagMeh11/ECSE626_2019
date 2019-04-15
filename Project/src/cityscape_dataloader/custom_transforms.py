import torch
import numpy as np

class Project(object):
    """
    Given a projection mapping (i.e. a dict) and an input tensor, this transform replaces
    all values in the tensor that equal a key in the mapping with the value corresponding to
    the key.
    """
    def __init__(self, projection):
        """
        Parameters
        ----------
        projection : dict
            The projection mapping.
        """
        super(Project, self).__init__()
        self.projection = dict(projection)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        output = torch.zeros_like(tensor)
        for source, target in self.projection.items():
            output[tensor == source] = target
        return output
 
    def __repr__(self):
        return self.__class__.__name__ + 'projection={}'.format(self.projection)

class DepthConversion(object):
    """
    Normalize a Disparity map tensor image .
    """

    def __init__(self, inplace=False):
        super(DepthConversion, self).__init__()
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Disparity Map Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Disparity Map Tensor image.
        """
        tensor = tensor.float() / 128.
        tensor = torch.round(tensor)
        return ( tensor.float() - 1. ) / (256.)

    def __repr__(self):
        return self.__class__.__name__


class PIL_To_Tensor(object):
    """
    Converts PIL image to tensor 
    """

    def __init__(self, inplace=False):
        super(PIL_To_Tensor, self).__init__()
        self.inplace = inplace

    def __call__(self, pic):
        """
        Args:
            pic (PIL image): PIL image
        returns:
            Tensor: PIL image converted to tensor
        """
        
        if pic.mode == 'RGB':
            img = torch.from_numpy(np.array(pic, np.float32))
            num_channel = 3
        elif pic.mode == 'L':
            img = torch.from_numpy(np.array(pic, np.uint8))
            num_channel = 1
        else:
        	img = torch.from_numpy(np.array(pic, np.int32))
        	num_channel = 1

        img = img.view(pic.size[1], pic.size[0], num_channel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img


class NormalizeRange(object):
    """Normalizes input by a constant."""
    def __init__(self, normalize_by=255.):
        """
        Parameters
        ----------
        normalize_by : float or int
            Scalar to normalize by.
        """
        super(NormalizeRange, self).__init__()
        self.normalize_by = float(normalize_by)

    def __call__(self, tensor):
        return tensor / self.normalize_by
