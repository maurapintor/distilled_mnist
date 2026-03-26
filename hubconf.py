dependencies = ['torch']


_WEIGHTS = {
    "student": "https://github.com/maurapintor/distilled_mnist/releases/download/checkpoints/mnist_distilled.pt",
    "teacher": "https://github.com/maurapintor/distilled_mnist/releases/download/checkpoints/mnist_teacher.pt",
}


def mnist_model(weights: str = None):
    import torch
    from model import MNISTModel
    model = MNISTModel()
    if weights is not None:
        if weights not in _WEIGHTS:
            raise ValueError(f"weights must be one of {list(_WEIGHTS)}, or None for untrained. Got '{weights}'.")
        state_dict = torch.hub.load_state_dict_from_url(
            _WEIGHTS[weights],
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    return model


