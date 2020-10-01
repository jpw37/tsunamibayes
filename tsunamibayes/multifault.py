from .utils import haversine

class MultiFault(BaseFault):
    """A class for data relating to multiple faults. This class should be used
    when two or more faults want to be examined simultaneously.
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
        self.num_faults = len(fault_objs)
        self.faults = []
        for fault in fault_objs:
            self.faults.append(fault)


    def _find_closest_fault(self, lat, lon):
        """For a given latitude and longitude, this internal function finds the
        closest fault (in self.faults) and returns the index of that fault.

        In the case that n latitudes and longitudes are given, it returns a
        length n array containing the indices of each of the closest faults.
        """
        dists_shape = self.num_faults if np.isscalar(lat) else (self.num_faults, len(lat))
        dists = np.zeros(dists_shape)
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
        # Find closest fault.

        # Call the depth function on that fault.
        raise NotImplementedError()


    def strike_map(self,lat,lon):
        raise NotImplementedError()


    def dip_map(self,lat,lon):
        raise NotImplementedError()
