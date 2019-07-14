class NetworkInputOutput(object):
    """Used for pulling out info at different layers. """

    def __init__(self, name):
        self.name = name


class NetworkHead(NetworkInputOutput):
    """Used for pulling out info at different layers. """

    def __init__(self, name, nodes, activation=None):
        super().__init__(name)
        self.activation = activation
        self.nodes = nodes


class NetworkInput(NetworkInputOutput):
    """Used for inputting inputs at different layers. """

    def __init__(self, name, layer_type, layer_num, tensor,
                 merge_mode=None, axis=-1):
        super().__init__(name)
        self.layer_type = layer_type
        self.layer_num = layer_num
        self.tensor = tensor
        self.merge_mode = merge_mode
        self.axis = axis
