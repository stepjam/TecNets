class Consumer(object):

    def consume(self, inputs):
        raise NotImplementedError('Must be overridden.')

    def get_summaries(self, prefix):
        del prefix
        return []

    def get_loss(self):
        return 0

    def verify(self, item):
        if item is None:
            raise RuntimeError(
                'Attempted to get summaries or loss before calling consume.')
        return item

    def get(self, dict, key):
        if key not in dict:
            raise KeyError("Tried to access consumer %s." % key)
        return dict[key]
