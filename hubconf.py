dependencies = ['torch']


def mnist_model(student: bool = True):
    from model import MNISTModel
    model = MNISTModel()
    if student:
        weigths_url = "https://github.com/maurapintor/distilled_mnist/releases/download/checkpoints/mnist_distilled.pt"
    else:
        weigths_url = "https://github.com/maurapintor/distilled_mnist/releases/download/checkpoints/mnist_teacher.pt"
    state_dict = torch.hub.load_state_dict_from_url(weigths_url)
    model.load_state_dict(state_dict)
    return model


