import os
from datetime import datetime


def make_dir(output):
    if os.path.exists(output) and os.path.isdir(output):
        print(f'output directory {output} already exists')
    else:
        os.mkdir(output, 777)
    time = datetime.strftime(datetime.utcnow(), '%Y-%m-%d_%H-%M-%S')
    os.mkdir(f'{output}{time}', 777)
    return output + time
