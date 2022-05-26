class ToBinary:
    """Convert [0, 255] tensors to binary tensors."""

    def __call__(self, image):
        image[image < 0.5] = 0.0
        image[image >= 0.5] = 1.0
        return image