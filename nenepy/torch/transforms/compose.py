from torchvision import transforms


class Compose(transforms.Compose):

    def __call__(self, *images):
        for t in self.transforms:
            images = t(*images)
        return images
