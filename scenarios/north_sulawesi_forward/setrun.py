from tsunamibayes.setrun import make_setrun
from tsunamibayes.utils import Config

config = Config()
config.read('defaults.cfg')
config.read('defaults.cfg')
setrun = make_setrun(config)

if __name__ == '__main__':
   rundata = setrun()
   rundata.write()