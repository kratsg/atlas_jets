''' Class definitions for dealing with ATLAS jets'''
from ROOT import TLorentzVector
import numpy as np
import root_numpy as rnp
try:
  import matplotlib.pyplot as pl
except:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as pl

class Jet(TLorentzVector, object):
  def __init__(self, *arg, **kwargs):
    '''Defines an offline jet'''
    """
      energy             : jet energy, E
      momentum_transverse: magnitude of momentum transverse
                                 to beam, mag(p)*sin(theta)
      mass               : invariant mass of jet
      pseudo-rapidity coordinates
        - eta            : -ln( tan[theta/2] )
        - phi            : polar angle in the transverse plane
        -- theta is angle between particle momentum and the beam axis
        -- see more: http://en.wikipedia.org/wiki/Pseudorapidity
      radius             : radius of jet (eta-phi coordinates)

      initialize it by Jet(TLorentzVector) or Jet({'Pt': #, 'm': #, 'eta': #, 'phi': #, ...})
    """

    copyVector = bool(len(arg) == 1)
    newVector  = bool(len(kwargs) >= 4)

    # require that exactly one of the sets of arguments are valid length
    if not(copyVector ^ newVector):
      raise ValueError('invalid number of arguments supplied')

    if copyVector:
      if isinstance(arg[0], TLorentzVector):
        TLorentzVector.__init__(self, arg[0])
      else:
        raise TypeError('expected a TLorentzVector')
    else:
      TLorentzVector.__init__(self)
      validKeys = ('pt','eta','phi','m')
      kwargs = dict((k.lower(), v) for k,v in kwargs.iteritems())
      if all(k in kwargs for k in validKeys):
        self.SetPtEtaPhiM(*(kwargs[k] for k in validKeys))
      else:
        raise ValueError('Missing specific keys to make new vector, {}'.format(validKeys))

    self._radius    = np.float(kwargs.get('radius', 1.0))

    self._tau       = np.array(kwargs.get('tau', [None,None,None]))
    self._split     = np.array(kwargs.get('split', [None,None,None]))
    self._subjetsPt = np.array(kwargs.get('subjetsPt', []))

  @property
  def coord(self):
    return (self.phi, self.eta)
  @property
  def pt(self):
    return self.Pt()
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
  def nsj(self):
    return self._subjetsPt.size
  @property
  def tau(self):
    return self._tau
  @property
  def split(self):
    return self._split
  @property
  def subjetsPt(self):
    return self._subjetsPt
  # serialize by returning a recarray
  @property
  def as_rec(self):
    # want to extend recarrays?
    #   np.array(b.as_rec.tolist() + c.tolist(), dtype=(b.as_rec.dtype.descr + c.dtype.descr) )
    # or if there are lists...
    #   np.array( [b.as_rec.tolist()[0] + c.tolist()[0] ], dtype=(b.as_rec.dtype.descr + c.dtype.descr) )
    datatype = ([('oJet.pt','float32'),\
                  ('oJet.eta','float32'),\
                  ('oJet.phi', 'float32'),\
                  ('oJet.m','float32'),\
                  ('oJet.rapidity','float32'),\
                  ('oJet.radius','float32'),\
                  ('oJet.nsj','int32'),\
                  ('oJet.tau','3float32'),\
                  ('oJet.split','3float32'),\
                  ('oJet.subjetsPt', 'object')])
    return np.array([(self.pt, self.eta, self.phi, self.m, self.rapidity, self.radius, self.nsj, self.tau, self.split, self.subjetsPt)], dtype=datatype )

  def __str__(self):
    if not hasattr(self,'_str'):
      self._str = ["oJet object"]
      self._str.append( "\t(phi,eta): ({:0.4f}, {:0.4f})".format(self.phi, self.eta) )
      self._str.append( "\tPt: {:0.2f} GeV".format(self.pt) )
      self._str.append( "\tm:  {:0.2f} GeV".format(self.m) )
      if np.any(self.tau):
        self._str.append( "\ttau: {}".format(map(lambda x: round(x,2), self.tau)) )
      if np.any(self.split):
        self._str.append( "\tsplit: {}".format(map(lambda x: round(x,2), self.split)) )
      if self.subjetsPt:
        self._str.append( "\tsubjetsPt: {}".format(map(lambda x: round(x,2), self.subjetsPt)) )
    return "\n".join(self._str)

class Event:
  # ToDo: rewrite Event[] to be a dictionary so we don't rely on ordering
  def __init__(self, event = []):
    self.jets = []
    # format comes in a list of jet data + subjet pt, identified by index
    jetData = event[:-1]
    subjetsPt = event[-1]/1000.
    for jetPt, jetM, jetEta, jetPhi, nsj, tau1, tau2, tau3, split12, split23, split34, subjets_index in zip(*jetData):
      # don't forget to scale from [MeV] -> [GeV]
      self.jets.append(Jet(Pt=jetPt/1000.,\
                            m=jetM/1000.,\
                            eta=jetEta,\
                            phi=jetPhi,\
                            nsj=nsj,\
                            tau=np.array([tau1,tau2,tau3]),\
                            split=np.array([split12,split23,split34]),\
                            subjetsPt=subjetsPt[subjets_index]))
    self.jets.sort(key=lambda jet: jet.pt, reverse=True)

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
    return "oEvent object with {:d} oJet objects".format(len(self.jets))
