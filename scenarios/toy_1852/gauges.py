from tsunamibayes import Gauge, dump_gauges
from tsunamibayes.utils import Config
import scipy.stats as stats

def build_gauges():
    """Creates gauge object for each observation point's data and appends each to a list.

    Returns
    -------
    gauges : (list) of Gauge objects
    """
    # Check if gauges have already been created
    if hasattr(build_gauges, 'gauges') and build_gauges.gauges is not None:
        return build_gauges.gauges

    config = Config()
    config.read('config_file.cfg')
    
    gauges = list()
    for ob in config.obs.values():
        name = ob['name']
        dists = dict()
        gauge_loc = 0
        gauge_scale = 0
        gauge_df = None
        for dist in ob['dists']:
            dist_type = dist[0]
            dist_name = dist[1]
            
            if dist_type == 'height':
                gauge_loc, gauge_scale = dist[-2:]
                
            if dist_name == 'norm':
                dists[dist_type] = stats.norm(loc=dist[2], scale=dist[3])
            elif dist_name == 'skewnorm':
                dists[dist_type] = stats.skewnorm(a=dist[2], loc=dist[3], scale=dist[4])
            elif dist_name == 'chi':
                gauge_df = dist[2]
                dists[dist_type] = stats.chi(df=dist[2], loc=dist[3], scale=dist[4])
        
        gauge = Gauge(name, dists)
        gauge.lat = ob['lat']
        gauge.lon = ob['lon']
        gauge.loc = gauge_loc
        gauge.scale = gauge_scale
        
        if any(dist[0] == 'inundation' for dist in ob['dists']):
            gauge.beta = ob['beta']
            gauge.n = ob['n']
            
        if gauge_df is not None:
            gauge.df = gauge_df
            
        gauges.append(gauge)

    # Cache the gauges
    build_gauges.gauges = gauges
    return gauges

if __name__ == "__main__":
    """Builds the scenario's gauges and stores the data in either a default file,
    or a file specified by the user in the command line."""
    from sys import argv

    if len(argv) == 1:
        gauges_path = 'data/gauges.json'
    else:
        gauges_path = argv[1]

    dump_gauges(build_gauges(), gauges_path)
