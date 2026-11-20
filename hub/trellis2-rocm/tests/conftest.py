import torch


def pytest_report_header(config):
    from ._load import k
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    return f"kernel: {k.__name__} from {k.__file__} | torch {torch.__version__} | {dev}"
