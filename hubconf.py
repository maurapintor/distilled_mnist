dependencies = ['torch']


def mnist_model(student: Bool = True):
    from model import MNISTModel
    model = MNISTModel()
    if student:
        weigths_url = ""
    else:
        weigths_url = ""
    state_dict = torch.hub.load_state_dict_from_url(weigths_url)
    model.load_state_dict(state_dict)
    return model


