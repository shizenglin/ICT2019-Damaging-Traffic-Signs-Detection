from torchvision import transforms
import torchvision.transforms.functional as F


class ResizeWithPad(object):

    def __init__(self, size):
        super(ResizeWithPad, self).__init__()
        self._resize = transforms.Resize(size)
        self._size = size

    def __call__(self, img):
        W, H = img.size
        if H > W:
            pad = (H - W) // 2
            padding = (pad, 0)
        else:
            pad = (W - H) // 2
            padding = (0, pad)
        img = F.pad(img, padding, 0, 'constant')
        return self._resize(img)


def ResizeNoAspectRatio(size):
    return transforms.Resize((size, size))
