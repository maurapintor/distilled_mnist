dependencies = ['torch']


def mnist_model(student: bool = True):
    import torch
    from model import MNISTModel
    model = MNISTModel()
    if student:
        weights_url = "https://github.com/maurapintor/distilled_mnist/releases/download/checkpoints/mnist_distilled.pt"
    else:
        weights_url = "https://github.com/maurapintor/distilled_mnist/releases/download/checkpoints/mnist_teacher.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        weights_url,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model


