import yaml
from pathlib import Path
import os

path = r'C:\BANGLV\ray_serving\config\config.yml'
if os.path.isfile(path):
    conf = yaml.safe_load(Path(path).read_text())
else:
    raise FileExistsError