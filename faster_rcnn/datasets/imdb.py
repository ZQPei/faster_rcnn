
class Imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._class_to_ind = {}
        self._image_index = []

    @property
    def name(self):
        return self._name
    @property
    def classes(self):
        return self._classes
    @property
    def num_classes(self):
        return self._num_classes
    @property
    def class_to_ind(self):
        return self._class_to_ind
    @property
    def image_index(self):
        return self._image_index