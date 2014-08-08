from ROOT import TLorentzVector

class T(TLorentzVector, object):
  # arguments and keyword arguments
  def __init__(self, *arg, **kwargs):
    copyVector = bool(len(arg) == 1)
    newVector  = bool(len(kwargs) == 4)

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

  @property
  def Pt(self):
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

a = TLorentzVector()
a.SetPtEtaPhiM(100, 0.5, 1.8, 100)

b = T(a)
c = T(Pt=100, eta=0.5, phi=1.8, M=100)

print bool(a == b == c)
