"""Module for defining interface to transient models.
"""

from abc import ABC, abstractmethod
from astropy import units as u
import numpy as np

from desiutil.log import get_logger, DEBUG

# Hide sncosmo import from the module.
try:
    import sncosmo
    log = get_logger(DEBUG)
    log.info('Enabling sncosmo models.')
    use_sncosmo = True
except ImportError as e:
    log = get_logger(DEBUG)
    log.warning('{}; disabling sncosmo models.'.format(e))
    use_sncosmo = False


class Transient(ABC):
    """Abstract base class to enforce interface for transient flux models."""

    def __init__(self, modelname, modeltype):

        self.model = modelname
        self.type = modeltype

        self.hostratio = 1.
        self.phase = 0.*u.day

    @abstractmethod
    def minwave(self):
        pass

    @abstractmethod
    def maxwave(self):
        pass

    @abstractmethod
    def mintime(self):
        pass

    @abstractmethod
    def maxtime(self):
        pass

    @abstractmethod
    def set_model_pars(modelpars):
        pass

    @abstractmethod
    def flux(self, t, wl):
        pass

if use_sncosmo:

    class Supernova(Transient):

        def __init__(self, modelname, modeltype, modelpars):
            """Initialize a built-in supernova model from the sncosmo package.

            Parameters
            ----------
            modelname : str
                Name of the model.
            modeltype : str
                Type or class of the model [Ia, IIP, ...].
            modelpars : dict
                Parameters used to initialize the model.
            """
            super().__init__(modelname, modeltype)

            # In sncosmo, some models have t0=tmax, and others have t0=0.
            # These lines ensure that for our purposes t0=tmax=0 for all models.
            self.t0 = modelpars['t0'] * u.day
            modelpars['t0'] = 0.

            self.snmodel = sncosmo.Model(self.model)
            self.set_model_pars(modelpars)

        def minwave(self):
            """Return minimum wavelength stored in model."""
            return self.snmodel.minwave() * u.Angstrom

        def maxwave(self):
            """Return maximum wavelength stored in model."""
            return self.snmodel.maxwave() * u.Angstrom

        def mintime(self):
            """Return minimum time used in model (peak light at t=0)."""
            return self.snmodel.mintime() * u.day - self.t0

        def maxtime(self):
            """Return maximum time used in model (peak light at t=0)."""
            return self.snmodel.maxtime() * u.day - self.t0

        def set_model_pars(self, modelpars):
            """Set sncosmo model parameters.

            Parameters
            ----------
            modelpars : dict
                Parameters used to initialize the internal model.
            """
            self.snmodel.set(**modelpars)

        def flux(self, t, wl):
            """Return flux vs wavelength at a given time t.

            Parameters
            ----------
            t : float or astropy.units.quantity.Quantity
                Time of observation, with t=0 representing max light.
            wl : list or ndarray
                Wavelength array to compute the flux.

            Returns
            -------
            flux : list or ndarray
                Normalized flux array as a function of wavelength.
            """
            # Time should be expressed w.r.t. maximum, in days.
            if type(t) is u.quantity.Quantity:
                self.phase = t
            else:
                self.phase = t * u.day

            time_ = (self.phase + self.t0).to('day').value

            # Convert wavelength to angstroms.
            wave_ = wl.to('Angstrom').value if type(wl) is u.quantity.Quantity else wl
            flux = self.snmodel.flux(time_, wl)
            return flux / np.sum(flux)


    class SupernovaBuilder:
        """A class which can build a Supernova. This allows the TransientModels
        object registry to register the model without instantiating it until it's
        needed. This is handy because some models take time and memory to
        instantiate.
        """

        def __init__(self):
            self._instance = None

        def __call__(self, modelpars):
            """Instantiate a Supernova using a list of modelpars.

            Parameters
            ----------
            modelpars : dict
                Parameters needed to create a Supernova (name, type, params).

            Returns
            -------
            instance : Supernova
            """
            if self._instance is None:
                self._instance = Supernova(**modelpars)
            return self._instance


class TabularModel(Transient):

    def __init__(self, modelname, modeltype, filename, filefmt):
        """Initialize a model from tabular data in an external file.

        Parameters
        ----------
        modelname : str
            Name of the model.
        modeltype : str
            Type or class of the model [TDE, AGN, ...].
        filename : str
            File with columns of wavelength and flux.
        filefmt : str
            File format (ascii, csv, fits, hdf5, ...).
        """
        super().__init__(modelname, modeltype)

        from astropy.table import Table
        data = Table.read(filename, format=filefmt, names=['wavelength','flux'])
        self.wave_ = data['wavelength'].data
        self.flux_ = data['flux'].data
        
        from scipy.interpolate import PchipInterpolator
        self.fvsw_ = PchipInterpolator(self.wave_, self.flux_)

    def minwave(self):
        """Return minimum wavelength stored in model."""
        return self.wave_[0] * u.Angstrom

    def maxwave(self):
        """Return maximum wavelength stored in model."""
        return self.wave_[-1] * u.Angstrom

    def mintime(self):
        """Return minimum time used in model (peak light at t=0)."""
        return 0 * u.day

    def maxtime(self):
        """Return maximum time used in model (peak light at t=0)."""
        return 1 * u.day

    def set_model_pars(self, modelpars):
        """Set model parameters.

        Parameters
        ----------
        modelpars : dict
            Parameters used to initialize the internal model.
        """
        pass

    def flux(self, t, wl):
        """Return flux vs wavelength at a given time t.

        Parameters
        ----------
        t : float or astropy.units.quantity.Quantity
            Time of observation, with t=0 representing max light.
        wl : list or ndarray
            Wavelength array to compute the flux.

        Returns
        -------
        flux : list or ndarray
            Normalized flux array as a function of wavelength.
        """
        # Convert wavelength to angstroms.
        wave_ = wl.to('Angstrom').value if type(wl) is u.quantity.Quantity else wl
        flux = self.fvsw_(wave_)
        return flux / np.sum(flux)


class TabularModelBuilder:
    """A class which can build a TabularModel. This allows the TransientModels
    object registry to register the model without instantiating it until it's
    needed. This is handy because some models take time and memory to
    instantiate.
    """

    def __init__(self):
        self._instance = None

    def __call__(self, modelpars):
        """Instantiate a TabularModel using a list of modelpars.

        Parameters
        ----------
        modelpars : dict
            Parameters needed to create a TabularModel (modelname, modeltype, filename, filefmt).

        Returns
        -------
        instance : TabularModel
        """
        if self._instance is None:
            self._instance = TabularModel(**modelpars)
        return self._instance


class TransientModels:

    def __init__(self):
        """Create a registry of transient model builder classes, model types,
        and model parameters.
        """
        self._builders = {}
        self._modelpars = {}
        self._types = {}

    def register_builder(self, modelpars, builder):
        """Register a model builder.
        
        Parameters
        ----------
        modelpars : dict
            Dictionary of model parameters (type, name, params).
        builder :
            A Transient builder class which instantiates a Transient.
        """
        modtype, modname = modelpars['modeltype'], modelpars['modelname']

        if modtype in self._types:
            self._types[modtype].append(modname)
        else:
            self._types[modtype] = [modname]

        self._builders[modname] = builder
        self._modelpars[modname] = modelpars

    def get_model(self, modelname):
        """Given a model name, returns a Transient using its builder.

        Parameters
        ----------
        modelname : str
            Name of registered Transient model.

        Returns
        -------
        instance : Transient
            Instance of a registered transient.
        """
        builder = self._builders.get(modelname)
        modelpars = self._modelpars.get(modelname)
        if not builder:
            raise ValueError(modelname)
        return builder(modelpars)

    def get_type_dict(self):
        """Return a dictionary of registered model types.

        Returns
        -------
        types : dict
            Dictionary of types and models.
        """
        return self._types

    def get_type(self, modeltype):
        """Given a Transient type, randomly return a registered model of that
        type.

        Parameters
        ----------
        modeltype : str
            Transient type (Ia, Ib, IIP, ...).

        Returns
        -------
        instance : Transient
            A registered Transient of the requested type.
        """
        mtype = self._types.get(modeltype)
        if not mtype:
            raise ValueError(modeltype)
        mname = np.random.choice(mtype)
        return self.get_model(mname)

    def __str__(self):
        """A list of registered transient types and model names.

        Returns
        -------
        repr : str
            Representation of registered model types and names.
        """
        s = []
        for t, models in self._types.items():
            s.append('- {}'.format(t))
            for m in models:
                s.append('  + {}'.format(m))
        return '\n'.join(s)


transients = TransientModels()

# Set up sncosmo models.
if use_sncosmo:

    # Register SN Ia models
    transients.register_builder({ 'modelname': 'hsiao',
                                  'modeltype': 'Ia', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'nugent-sn1a',
                                  'modeltype': 'Ia', 
                                  'modelpars': {'z':0., 't0':20., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'nugent-sn91t',
                                  'modeltype': 'Ia', 
                                  'modelpars': {'z':0., 't0':20., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'nugent-sn91bg',
                                  'modeltype': 'Ia', 
                                  'modelpars': {'z':0., 't0':15., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'salt2-extended',
                                  'modeltype': 'Ia', 
                                  'modelpars': {'z':0., 't0':0., 'x0':1., 'x1':0., 'c':0.} },
                                SupernovaBuilder())

    # Register SN Ib models
    transients.register_builder({ 'modelname': 's11-2005hl',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':-5., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 's11-2005hm',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':5., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 's11-2006jo',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2004gv',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006ep',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007y',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2004ib',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2005hm',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007nc',
                                  'modeltype': 'Ib', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN Ib/c models
    transients.register_builder({ 'modelname': 'nugent-sn1bc',
                                  'modeltype': 'Ib/c', 
                                  'modelpars': {'z':0., 't0':20., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN Ic models
    transients.register_builder({ 'modelname': 's11-2006fo',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2004fe',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2004gq',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-sdss004012',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006fo',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-sdss014475',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006lc',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-04d1la',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-04d4jv',
                                  'modeltype': 'Ic', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN IIn models
    transients.register_builder({ 'modelname': 'nugent-sn2n',
                                  'modeltype': 'IIn', 
                                  'modelpars': {'z':0., 't0':20., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006ez',
                                  'modeltype': 'IIn', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006ix',
                                  'modeltype': 'IIn', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN IIP models
    transients.register_builder({ 'modelname': 'nugent-sn2p',
                                  'modeltype': 'IIP', 
                                  'modelpars': {'z':0., 't0':20., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 's11-2005lc',
                                  'modeltype': 'IIP', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 's11-2005gi',
                                  'modeltype': 'IIP', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 's11-2006jl',
                                  'modeltype': 'IIP', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2004hx',
                                  'modeltype': 'IIP', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2005gi',
                                  'modeltype': 'IIP', 
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006gq',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006kn',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006jl',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006iw',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006kv',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2006ns',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007iz',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007nr',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007kw',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007ky',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007lj',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007lb',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007ll',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007nw',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007ld',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007md',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007lz',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007lx',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007og',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007nv',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    transients.register_builder({ 'modelname': 'snana-2007pg',
                                  'modeltype': 'IIP',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN IIL
    transients.register_builder({ 'modelname': 'nugent-sn2l',
                                  'modeltype': 'IIL',
                                  'modelpars': {'z':0., 't0':12., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN IIL/P
    transients.register_builder({ 'modelname': 's11-2004hx',
                                  'modeltype': 'IIL/P',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())

    # Register SN II-pec
    transients.register_builder({ 'modelname': 'snana-2007ms',
                                  'modeltype': 'II-pec',
                                  'modelpars': {'z':0., 't0':0., 'amplitude':1.} },
                                SupernovaBuilder())
