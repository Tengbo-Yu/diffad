from mmdet.utils import get_git_hash
from mmdet import __version__

def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['mmdet'] = __version__
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
