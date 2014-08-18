''' Class definitions for dealing with ATLAS jets using towers'''
from ROOT import TLorentzVector
import root_numpy as rnp
import numpy as np
try:
  import matplotlib.pyplot as pl
except:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as pl

#np.seterr(divide='ignore', invalid='ignore') #numpy complains on line 271 about Tower::rho calculation

class SeedFilter:
  def __init__(self, et = 0., n = 20.):
    self._et = np.float(et)
    self._n  = np.int(n)

  @property
  def et(self):
    return self._et
  @property
  def n(self):
    return self._n

  def __call__(self, towers):
    return [tower for tower in towers if tower.et > self.et][:self.n]

  def __str__(self):
    return "SeedFilter object\n\tat most {:d} seeds with Et > {:0.4f} GeV".format(self.n, self.et)

class Tower(TLorentzVector, object):
  def __init__(self, *arg, **kwargs):
    '''Defines a gTower'''
    """
      initialize it by passing in kwargs that contain all information
    """
    newVector  = bool(len(kwargs) == 6)

    # require that exactly one of the sets of arguments are valid length
    if not newVector:
      raise ValueError('invalid number of keyword arguments supplied')

    TLorentzVector.__init__(self)

    validKeys = ('et','etamin','etamax','phimin','phimax','num_cells')
    kwargs = dict((k.lower(), v) for k,v in kwargs.iteritems())
    if all(k in kwargs for k in validKeys):
      # set the center of the tower to the geometric center
      self.SetPtEtaPhiM(\
        kwargs['et'],\
        (kwargs['etamax'] + kwargs['etamin'])/2.0,\
        (kwargs['phimax'] + kwargs['phimin'])/2.0,\
        0.0)
    else:
      raise ValueError('Missing specific keys to make new vector, {}'.format(validKeys))
    self._etamax = kwargs['etamax']
    self._etamin = kwargs['etamin']
    self._phimax = kwargs['phimax']
    self._phimin = kwargs['phimin']
    self._num_cells = np.int(kwargs['num_cells'])

  @property
  def et(self):
    return self.Et()
  @property
  def eta(self):
    return self.Eta()
  @property
  def phi(self):
    return self.Phi()
  @property
  def m(self):
    return self.M()
  @property
  def num_cells(self):
    return self._num_cells
  @property
  def rapidity(self):
    return self.Rapidity()
  @property
  def area(self):
    return np.abs((self.etamax - self.etamin) * (self.phimax - self.phimin))
  @property
  def rho(self):
    return self.et / self.area
  @property
  def phimin(self):
    return self._phimin
  @property
  def phimax(self):
    return self._phimax
  @property
  def etamin(self):
    return self._etamin
  @property
  def etamax(self):
    return self._etamax
  @property
  def region(self):
    if -1.6 <= self.eta < 0.0:
      return 1
    elif 0.0 <= self.eta < 1.6:
      return 2
    elif -4.9 <= self.eta < -1.6:
      return 3
    elif 1.6 <= self.eta < 4.9:
      return 4

  # serialize by returning a recarray
  @property
  def as_rec(self):
    # want to extend recarrays?
    #   np.array(b.as_rec.tolist() + c.tolist(), dtype=(b.as_rec.dtype.descr + c.dtype.descr) )
    # or if there are lists...
    #   np.array( [b.as_rec.tolist()[0] + c.tolist()[0] ], dtype=(b.as_rec.dtype.descr + c.dtype.descr) )
    datatype = ([('tower.et','float32'),\
                ('tower.eta','float32'),\
                ('tower.phi', 'float32'),\
                ('tower.m','float32'),\
                ('tower.num_cells','int32'),\
                ('tower.rapidity','float32'),\
                ('tower.area','float32'),\
                ('tower.rho','float32')])
    return np.array([(self.et, self.eta, self.phi, self.m, self.num_cells, self.rapidity, self.area, self.rho)], dtype=datatype )

  def __str__(self):
    if not hasattr(self,'_str'):
      self._str = ["gTower object"]
      self._str.append( "\t(phi,eta): ({:0.4f}, {:0.4f})".format(self.phi, self.eta) )
      self._str.append( "\tEt: {:0.2f} GeV".format(self.et) )
      self._str.append( "\tm:  {:0.2f} GeV".format(self.m) )

      self._str.append( "\trho: {:0.2f}".format(self.rho) )
      self._str.append( "\tarea: {:0.2f}".format(self.area) )
      self._str.append( "\tnum_cells: {:d}".format(self.num_cells) )
    return "\n".join(self._str)


class Jet(TLorentzVector, object):
  def __init__(self, *arg, **kwargs):
    '''Defines a trigger jet'''
    """
      vector             : a TLorentzVector() defined from ROOT that contains
                              information about the Jet's 4-vector
      area               : jet area based on sum of gTower areas that made jet
      radius             : radius of jet (eta-phi coordinates)
      towers_around      : contains the top 3 gTowers
      seed               : contains the seed used for this jet

      initialize it by passing in a TLorentzVector() object plus kwargs that 
        contain area, radius, and towers_around
    """

    copyVector = bool(len(arg) == 1)
    newVector  = bool(len(kwargs) == 4)

    # require that exactly one of the sets of arguments are valid length
    if not(copyVector and newVector):
      raise ValueError('invalid number of arguments supplied')

    if isinstance(arg[0], TLorentzVector):
      TLorentzVector.__init__(self, arg[0])
    else:
      raise TypeError('expected a TLorentzVector')

    #TLorentzVector.__init__(self)
    validKeys = ('area','radius','towers','seed')
    kwargs = dict((k.lower(), v) for k,v in kwargs.iteritems())
    if all(k in kwargs for k in validKeys):
      self._area    = np.float(kwargs['area'])
      self._radius  = np.float(kwargs['radius'])
      self._towers  = np.array(kwargs['towers'])
      self._seed    = kwargs['seed']
    else:
      raise ValueError('Missing specific keys to make tJet object, {}'.format(validKeys))

  @property
  def coord(self):
    return (self.phi, self.eta)
  @property
  def et(self):
    return self.Et()
  @property
  def eta(self):
    return self.Eta()
  @property
  def phi(self):
    return self.Phi()
  @property
  def m(self):
    return self.M()
  @property
  def rapidity(self):
    return self.Rapidity()
  #these are extra arguments passed in
  @property
  def radius(self):
    return self._radius
  @property
  def area(self):
    return self._area
  @property
  def seed(self):
    return self._seed
  @property
  def towers(self):
    return self._towers
  @property
  def region(self):
    if -1.6 <= self.eta < 0.0:
      return 1
    elif 0.0 <= self.eta < 1.6:
      return 2
    elif -4.9 <= self.eta < -1.6:
      return 3
    elif 1.6 <= self.eta < 4.9:
      return 4
  # serialize by returning a recarray
  @property
  def as_rec(self):
    # want to extend recarrays?
    #   np.array(b.as_rec.tolist() + c.tolist(), dtype=(b.as_rec.dtype.descr + c.dtype.descr) )
    # or if there are lists...
    #   np.array( [b.as_rec.tolist()[0] + c.tolist()[0] ], dtype=(b.as_rec.dtype.descr + c.dtype.descr) )
    datatype = ([('tJet.et','float32'),\
                  ('tJet.eta','float32'),\
                  ('tJet.phi', 'float32'),\
                  ('tJet.m','float32'),\
                  ('tJet.rapidity','float32'),\
                  ('tJet.radius','float32'),\
                  ('tJet.area','float32'),\
                  ('tJet.seed','object'),\
                  ('tJet.towers', '3object')])
    return np.array([(self.et, self.eta, self.phi, self.m, self.rapidity, self.radius, self.area, self.seed, self.towers)], dtype=datatype )

  def __str__(self):
    if not hasattr(self,'_str'):
      self._str = ["gJet object"]
      self._str.append( "\t(phi,eta): ({:0.4f}, {:0.4f})".format(self.phi, self.eta) )
      self._str.append( "\tEt: {:0.2f} GeV".format(self.et) )
      self._str.append( "\tm:  {:0.2f} GeV".format(self.m) )

      self._str.append( "\tradius: {:0.2f}".format(self.radius) )
      self._str.append( "\tarea: {:0.2f}".format(self.area) )
      self._str.append( "======SEED======" )
      self._str.append( str(self.seed) )
    return "\n".join(self._str)

class TowerEvent:
  def __init__(self, event = [], seed_filter = SeedFilter()):
    self.towers = []
    self.seed_filter = seed_filter
    for gTowerEt, gTowerNCells, gTowerEtaMin, gTowerEtaMax, gTowerPhiMin, gTowerPhiMax in zip(*event):
      # noisy gTowers do not need to be included
      self.towers.append(Tower(Et=gTowerEt/1000.,\
                               num_cells=gTowerNCells,\
                               etaMin=gTowerEtaMin,\
                               etaMax=gTowerEtaMax,\
                               phiMin=gTowerPhiMin,\
                               phiMax=gTowerPhiMax))
      self.towers.sort(key=lambda tower: tower.et, reverse=True)

  def towers_below(self, et):
    return [tower for tower in self.towers if tower.et < et]

  def towers_above(self, et):
    return [tower for tower in self.towers if tower.et > et]

  def towers_between(self, low, high):
    return [tower for tower in self.towers if high > tower.et > low]

  def filter_towers(self, towers=None):
    towers = towers or self.towers
    return self.seed_filter(towers)

  def get_event(self, radius=1.0, towers=None):
    towers = towers or self.towers
    self.__seeds_to_jet(radius=radius, towers=towers)
    return self.event

  def __seeds_to_jet(self, radius=1.0, towers=None):
    towers = towers or self.towers
    jets = []
    for seed in self.filter_towers(towers=towers):
      jets.append(self.__seed_to_jet(seed, radius=radius, towers=towers))
    self.event = Event(jets=jets)

  def __seed_to_jet(self, seed, radius=1.0, towers=None):
    towers = towers or self.towers
    # note: each tower has m=0, so E = p, ET = Pt
    l = seed
    jet_area = seed.area
    towers_around = self.__towers_around(seed, radius=radius, towers=towers)
    for tower in towers_around:
      #radius = 1.0
      #normalization = 2. * np.pi * radius**2. * erf( 0.92 * (2.**-0.5) )**2.
      #exponential = np.exp(-( (seed.phi - tower.phi)**2./(2. * (radius**2.)) + (seed.eta - tower.eta)**2./(2.*(radius**2.)) ))
      #towerTLorentzVector.SetPtEtaPhiM(tower.E/np.cosh(tower.eta) * exponential/normalization, tower.eta, tower.phi, 0.0)

      # don't do l += tower, fucking doesn't copy fucking fuck fucking
      l = l + tower
      jet_area += tower.area
    return Jet(l, area=jet_area, radius=radius, seed=seed, towers=towers_around[:3])

  def __towers_around(self, seed, radius=1.0, towers=None):
    towers = towers or self.towers
    def distance_between(a,b):
      #a = (phi, eta); b = (phi, eta)
      delta = np.abs(a-b)
      delta = np.array( [2*np.pi - delta[0] if delta[0] > np.pi else delta[0], delta[1]] ) #deal with periodicity in phi
      return np.sqrt((delta**2.).sum(axis=-1))

    return [tower for tower in towers if distance_between( np.array([tower.phi, tower.eta]), np.array([seed.phi, seed.eta]) ) <= radius and tower != seed]

  def __iter__(self):
    # initialize to start of list
    self.iter_index = -1
    # `return self` to use `next()`
    return self

  def next(self):
    self.iter_index += 1
    if self.iter_index == len(self.towers):
      raise StopIteration
    return self.towers[self.iter_index]

  def __getitem__(self, index):
    return self.towers[index]

  def __str__(self):
    return "TowerEvent object with %d Tower objects" % (len(self.towers))

class Event:
  def __init__(self, jets = []):
    self.jets = jets
    self.jets.sort(key=lambda jet: jet.Pt, reverse=True)

  def __iter__(self):
    # initialize to start of list
    self.iter_index = -1
    # `return self` to use `next()`
    return self

  def next(self):
    self.iter_index += 1
    if self.iter_index == len(self.jets):
      raise StopIteration
    return self.jets[self.iter_index]

  def __getitem__(self, index):
    if isinstance( index, ( int, long ) ):
      return self.jets[index]
    elif index in self.jets[0]:
      return np.array([jet[index] for jet in self.jets])
    else:
      raise ValueError('Unclear what index is: {}, {}'.format(index, index.__class__))

  def __str__(self):
    return "tEvent object with {:d} tJet objects".format(len(self.jets))
