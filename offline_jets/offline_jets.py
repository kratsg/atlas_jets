''' Class definitions for dealing with ATLAS jets'''
from ROOT import TLorentzVector
import root_numpy as rnp
import numpy as np
try:
  import matplotlib.pyplot as pl
except:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as pl

def erf(x):
  '''returns the error function evaluated at x'''
  # does not necessarily need SciPy working, but less accurate without
  try:
    import scipy.special
    return scipy.special.erf(x)
  except ImportError:
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = np.fabs(x)
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def circularRegion(cell_jetcoord, cell_radius, cell_coord):
  '''determines if `cell_coord` is within `cell_radius` of `cell_jetcoord`'''
  diff = cell_jetcoord - cell_coord
  distance = np.sqrt(diff[0]**2. + diff[1]**2.)
  if distance <= cell_radius:
    return True
  else:
    return False

# all coordinates are in (phi, eta) pairs always
#     and are converted to (cell_x, cell_y) at runtime
class Grid:
  def __init__(self,\
               domain                = np.array([[-3.2,3.2],[0.0,3.2]]),\
               cell_resolution      = 0.20,\
               recon_algo            = 'gaussian'):
    '''Generates a grid of zeros based on size of domain and resolution'''
    """
        - domain: [ (phi_min, phi_max) , (eta_min, eta_max) ]
        - resolution
          - cell: how big a single cell is in eta-phi
        - recon_algo: which jet reconstruction algorithm to use
          - uniform  (implemented)
          - gaussian (implemented)
          - dipole   (not implemented)
          - j3       (not implemented)
    """
    self.domain                = np.array(domain).astype(float)
    self.cell_resolution      = np.float(cell_resolution)
    self.grid = np.zeros(self.phieta2cell(self.domain[:,1])).astype(float)
    valid_algorithms = ['uniform', 'gaussian', 'dipole', 'j3']
    if recon_algo not in valid_algorithms:
      raise ValueError('%s is not a valid algorithm. Choose from %s' % (recon_algo, valid_algorithms))
    self.recon_algo = recon_algo

  def phieta2cell(self, phieta_coord):
    '''Converts (phi,eta) -> (cell_x, cell_y)'''
    cell_coord = (phieta_coord - self.domain[:,0])/self.cell_resolution
    return np.round(cell_coord).astype(int)

  def cell2phieta(self, cell_coord):
    '''Converts (cell_x, cell_y) -> rounded(phi, eta)'''
    return self.cell_resolution * cell_coord + self.domain[:,0]

  def boundary_conditions(self, cell_coord):
    '''Checks if cell_coord is outside of Grid'''
    # at the moment, this boundary condition defined for eta
    #   because phi is periodic
    if 0 <= cell_coord[1] < self.grid.shape[1]:
      return True
    else:
      return False

  def add_event(self, event):
    for jet in event:
      self.add_jet(jet)

  def add_jet(self, jet):
    '''add a single `jet` to the current grid'''
    for cell_coord, fractional_energy in self.__generate_mesh(jet):
      try:
        if cell_coord[0] >= self.grid.shape[0]:
          cell_coord = (cell_coord[0] - self.grid.shape[0], cell_coord[1])
        self.grid[cell_coord] += fractional_energy
        jet.trigger_energy += fractional_energy
      except IndexError:
        # we should NEVER see this gorram error
        # -- the reason is that we filter out all inappropriate eta coordinates
        #        and then wrap around in the phi coordinates
        print "\t"*2, jet
        print "\t"*2, '-- jet cell_coord could not be added:', cell_coord

  def __jetdPt(self, cell_jetcoord, cell_radius, jet_energy, cell_coord):
    '''return the fractional energy at `cell_coord` of a `jet_energy` GeV jet centered at `cell_jetcoord` of radius `cell_radius`'''
    if self.recon_algo == 'uniform':
      # uniform energy is calculated in self.__generate_mesh due to efficiency concerns
      raise Exception('should not be calling this function when self.recon_algo == \'uniform\'')
      return false
    elif self.recon_algo == 'gaussian':
      return self.__gaussian2D(cell_jetcoord, cell_radius, jet_energy, cell_coord)

  def __gaussian2D(self, cell_mu, cell_radius, amplitude, cell_coord):
    '''return the 2D gaussian(mu, sigma) evaluated at coord'''
    # normalization factor, 500 GeV outputs 476.275
    # default is 2 * pi * sx * sy * amplitude
    normalization = 2. * np.pi * cell_radius**2. * erf( 0.92 * (2.**-0.5) )**2.
    exponential = np.exp(-( (cell_mu[0] - cell_coord[0])**2./(2. * (cell_radius**2.)) + (cell_mu[1] - cell_coord[1])**2./(2.*(cell_radius**2.)) ))
    return amplitude*exponential/normalization

  def __generate_mesh(self, jet):
    '''return the 2D mesh generator of `cell_coords` for a `jet` to add to grid'''
    # convert to cell coordinates and deal with grid
    cell_jetcoord = self.phieta2cell(jet.coord)
    cell_radius = jet.radius/self.cell_resolution
    # what we define as the jet energy for `self.__jetdPt`
    jet_energy = jet.Pt
    # always start with a square mesh
    square_mesh_coords = self.__square_mesh(cell_jetcoord, cell_radius)
    if self.recon_algo == 'uniform':
      uniform_jetdPt = jet_energy/(square_mesh_coords.size/2.)
      mesh = ([tuple(cell_coord), uniform_jetdPt] for cell_coord in square_mesh_coords if self.boundary_conditions(cell_coord) )
    elif self.recon_algo == 'gaussian':
      mesh = ([tuple(cell_coord), self.__jetdPt(cell_jetcoord, cell_radius, jet_energy, cell_coord)] for cell_coord in square_mesh_coords if self.boundary_conditions(cell_coord)&circularRegion(cell_jetcoord, cell_radius, cell_coord) )
    return mesh

  def __square_mesh(self, center, radius):
    '''generates a square meshgrid of points for center and sidelength 2*radius'''
    i,j = np.meshgrid( np.arange(center[0] - radius, center[0]+radius+1), np.arange(center[1] - radius, center[1]+radius+1) )
    return np.transpose([i.reshape(-1), j.reshape(-1)]).astype(int)

  def __make_plot(self, title='Grid Plot'):
    '''Creates a figure of the current grid'''
    fig = pl.figure()
    # plot the grid
    pl.imshow(self.grid, cmap = pl.cm.jet)
    # x-axis is phi, y-axis is eta
    #xticks_loc = pl.axes().xaxis.get_majorticklocs()
    #yticks_loc = pl.axes().yaxis.get_majorticklocs()
    plotTopLeft  = self.phieta2cell((-3.2,-4.9))
    plotBotRight = self.phieta2cell((3.2,4.9))
    plot_resolution = 0.2*2
    tickMarks = plot_resolution/self.cell_resolution
    xticks_loc = np.arange(plotTopLeft[1],plotBotRight[1] + 1,2*tickMarks)
    yticks_loc = np.arange(plotTopLeft[0],plotBotRight[0] + 1,tickMarks)
    # make labels
    pl.xlabel('$\eta$')
    pl.ylabel('$\phi$')
    pl.title(title)
    # transform labels from cell coords to phi-eta coords
    #xticks_label = xticks_loc * self.cell_resolution + self.domain[1,0]
    #yticks_label = yticks_loc * self.cell_resolution + self.domain[0,0]
    xticks_label = ["%0.1f" % i for i in np.arange(-4.9,4.9 + 2*plot_resolution,2*plot_resolution)]
    yticks_label = ["%0.1f" % i for i in np.arange(-3.2,3.2 + plot_resolution,plot_resolution)]
    # add in 0 by hardcoding
    #xticks_loc = np.append(xticks_loc,0)
    #xticks_label = np.append(xticks_label,'0')
    # set the labels
    pl.xticks(xticks_loc, xticks_label)
    pl.yticks(yticks_loc, yticks_label)
    '''fuck the colorbar. it's very non-descriptive with a grid'''
    return fig

  def show(self, title='Grid Plot'):
    '''Show an image of the current grid'''
    fig = self.__make_plot(title)
    fig.show()

  def save(self, title='Grid Plot', filename='output.png'):
    '''Save an image of the current grid to file'''
    fig = self.__make_plot(title)
    fig.savefig(filename)

  def __str__(self):
    return "Grid object:\n\tPhi: %s\n\tEta: %s\n\tResolution: %0.2f" % (self.domain[0], self.domain[1], self.cell_resolution)
    
class Jet(TLorentzVector, object):
  def __init__(self, *arg, **kwargs):
    '''Defines a jet'''
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

    self._nsj       = np.int(  kwargs.get('nsj', 0))
    self._tau       = np.array(kwargs.get('tau', []))
    self._split     = np.array(kwargs.get('split', []))
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
  def r(self):
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

  def __str__(self):
    if not hasattr(self,'_str'):
      self._str = ["oJet object"]
      self._str.append( "\t(phi,eta): ({:0.4f}, {:0.4f})".format(self.phi, self.eta) )
      self._str.append( "\tPt: {:0.2f} GeV".format(self.pt) )
      self._str.append( "\tm:  {:0.2f} GeV".format(self.m) )
      if self.tau:
        self._str.append( "\ttau: {}".format(map(lambda x: round(x,2), self.tau)) )
      if self.split:
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
    for jetE, jetPt, jetM, jetEta, jetPhi, nsj, tau1, tau2, tau3, split12, split23, split34, subjets_index in zip(*jetData):
      # don't forget to scale from [MeV] -> [GeV]
      self.jets.append(Jet({'Pt': jetPt/1000.,\
                            'm':  jetM/1000.,\
                            'eta':jetEta,\
                            'phi':jetPhi,\
                            'nsj':nsj,\
                            'tau':np.array([tau1,tau2,tau3]),\
                            'split':np.array([split12,split23,split34]),\
                            'subjetsPt':subjetsPt[subjets_index]}))
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
    else if index in self.jets[0]:
      return np.array([jet[index] for jet in self.jets])
    else:
      raise ValueError('Unclear what index is: {}, {}'.format(index, index.__class__))

  def __str__(self):
    return "Event object with {:d} Jet objects".format(len(self.jets))
