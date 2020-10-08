import numpy as np
from .utils import haversine
from .fault import BaseFault

class MultiFault(BaseFault):
    """A class for data relating to multiple faults. This class should be used
    when two or more faults want to be examined simultaneously (i.e. the fault
    that originated the earthquake is unknown).
    """
    def __init__(self, fault_objs, bounds):
        """Creates and initializes the MultiFault object with specified lat/lon
        bounds and any number of individual faults contained within it.

        Parameters
        ----------
        fault_objs : list of GridFault or ReferenceCurveFault objects
            These are the faults that will be contained and used by MultiFault.
        bounds : dict
            The dictionary of the upper and lower limits for latitude/longitude
            for the model. Keys are 'lon_min','lon_max','lat_min', 'lat_max',
            with associated (float) values.
        """
        super().__init__(bounds)
        self.faults = fault_objs


    def _find_closest_fault(self, lat, lon):
        """For a given latitude and longitude, this internal function finds the
        closest fault (in self.faults) and returns the index of that fault.

        In the case that n latitudes and longitudes are given, it returns a
        length n array containing the indices of each of the closest faults.
        """
        dists_shape = self.num_faults if np.isscalar(lat) else (self.num_faults, len(lat))
        dists = np.empty(dists_shape)
        for j,fault in enumerate(self.faults):
            if hasattr(fault, 'distance'):
                dists[j] = fault.distance(lat,lon)
            else:
                dists[j] = self._distance(j, lat, lon)
        return np.min(dists,axis=0)


    def _distance(self, fault_idx, lat, lon):
        """Compute the distance from (lat, lon) to self.faults[fault_idx]."""
        fault = self.faults[fault_idx]
        distances = haversine(lat,lon,fault.latpts,fault.lonpts)
        return distances.min()


    def depth_map(self,lat,lon):
        """Computes the depth given latitude and longitude coordinates."""
        depth = np.empty(np.size(lat))

        # Find closest fault(s).
        fault_indices = self._find_closest_fault(lat,lon)

        # Call the depth function on the fault(s).
        if np.isscalar(lat):
            depth = self.faults[fault_indices].depth_map(lat,lon)
        else:
            for j,fault in enumerate(self.faults):
                idx = np.where(fault_indices == j)[0]
                depth[idx] = fault.depth_map(lat[idx],lon[idx])
        return depth.reshape(lat.shape)


    def strike_map(self,lat,lon):
        """Computes the strike given latitude and longitude coordinates."""
        strike = np.empty(np.size(lat))

        # Find closest fault(s).
        fault_indices = self._find_closest_fault(lat,lon)

        # Call the depth function on the fault(s).
        if np.isscalar(lat):
            strike = self.faults[fault_indices].strike_map(lat,lon)
        else:
            for j,fault in enumerate(self.faults):
                idx = np.where(fault_indices == j)[0]
                strike[idx] = fault.strike_map(lat[idx],lon[idx])
        return strike.reshape(lat.shape)


    def dip_map(self,lat,lon):
        """Computes the dip given latitude and longitude coordinates."""
        dip = np.empty(np.size(lat))

        # Find closest fault(s).
        fault_indices = self._find_closest_fault(lat,lon)

        # Call the depth function on the fault(s).
        if np.isscalar(lat):
            dip = self.faults[fault_indices].dip_map(lat,lon)
        else:
            for j,fault in enumerate(self.faults):
                idx = np.where(fault_indices == j)[0]
                dip[idx] = fault.dip_map(lat[idx],lon[idx])
        return dip.reshape(lat.shape)
