''' Class definitions for dealing with ATLAS jets'''

#ROOT is needed to deal with rootfiles
import ROOT

#root_numpy is needed to read the rootfile
import root_numpy as rnp

# numpy and matplotlib (mpl) are used for computing and plotting 
import numpy as np
import matplotlib.pyplot as pl

def erf(x):
  '''returns the error function evaluated at x'''
  # does not necessarily need SciPy working, but less accurate without
  try:
    import scipy.special
    erf_loaded = True
  except ImportError:
    erf_loaded = False
  # use SciPy if installed
  if erf_loaded:
    return scipy.special.erf(x)
  else:
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
    jet_energy = jet.pT
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
    pl.imshow(self.grid.T, cmap = pl.cm.spectral)
    # x-axis is phi, y-axis is eta
    xticks_loc = pl.axes().xaxis.get_majorticklocs()
    yticks_loc = pl.axes().yaxis.get_majorticklocs()
    # make labels
    pl.xlabel('$\phi$')
    pl.ylabel('$\eta$')
    pl.title(title)
    # transform labels from cell coords to phi-eta coords
    xticks_label = xticks_loc * self.cell_resolution + self.domain[0,0]
    yticks_label = yticks_loc * self.cell_resolution + self.domain[1,0]
    pl.xticks(xticks_loc, xticks_label)
    pl.yticks(yticks_loc, yticks_label)
    # set the colorbar
    cbar = pl.colorbar(pad=0.2)
    cbar.set_label('pT (GeV)')
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
    
class Jet:
  def __init__(self,\
               inputThresh    = 200.,\
               triggerThresh  = 150.,\
               E              = 0.0,\
               pT             = 0.0,\
               m              = 0.0,\
               eta            = 0.0,\
               phi            = 0.0,\
               radius         = 1.0,\
               input_energy   = 0.0,\
               trigger_energy = 0.0):
    '''Defines a jet'''
    """
      thresholds
        - input          : in GeV, jet energy for input
        - trigger        : in GeV, jet energy for trigger
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
      input_energy       : the input energy of the jet
      trigger_energy     : amount of energy actually recorded on the grid
    """
    self.inputThresh    = np.float(inputThresh)
    self.triggerThresh  = np.float(triggerThresh)
    self.E              = np.float(E)
    self.pT             = np.float(pT)
    self.m              = np.float(m)
    self.phi            = np.float(phi)
    self.eta            = np.float(eta)
    self.coord          = (self.phi, self.eta)
    self.radius         = np.float(radius)
    self.input_energy   = np.float(pT)
    self.trigger_energy = np.float(trigger_energy)

  def __str__(self):
    return "Jet object:\n\tPhi: %0.4f\n\tEta: %0.4f\n\tE: %0.2f (GeV)\n\tpT: %0.2f (GeV)\n\tm: %0.2f (GeV)\n\tInputted: %s\n\tTriggered: %s" % (self.phi, self.eta, self.E, self.pT, self.m, self.inputted(), self.triggered())

  def inputted(self):
    return self.input_energy > self.inputThresh

  def triggered(self):
    return self.trigger_energy > self.triggerThresh

class Events:
  def __init__(self, rootfile):
    self.rootfile = rootfile
    self.events   = []
    self.load()

  def load(self):
    # we want jet_AntiKt4LCTopo_ [E, pt, m, eta, phi] but not [n]
    indices = [self.rootfile.data.dtype.names.index(name) for name in self.rootfile.data.dtype.names if 'jet_AntiKt4LCTopo_' in name][1:6]
    self.events = [Event(event=[event[i] for i in indices]) for event in self.rootfile.data]
    print 'Loaded offline events.'

  def __iter__(self):
    # initialize to start of list
    self.iter_index = -1
    # `return self` to use `next()`
    return self

  def next(self):
    self.iter_index += 1
    if self.iter_index == len(self.events):
      raise StopIteration
    return self.events[self.iter_index]

  def __getitem__(self, index):
    return self.events[index]

  def __str__(self):
    return "Events object with %d Event objects" % len(self.events)
  
class Event:
  def __init__(self, event = []):
    self.jets = []
    # format generally comes as a tuple of 10 lists, each list
    #    is filled by that property for all jets like so
    #  ( [ jetE_0, jetE_1, jetE_2], [ jetPt_0, jetPt_1, jetPt_2 ], ...)
    for jetE, jetPt, jetM, jetEta, jetPhi in zip(*event):
      # don't forget to scale from [MeV] -> [GeV]
      self.jets.append(Jet(E=jetE/1000.,\
                           pT=jetPt/1000.,\
                           m=jetM/1000.,\
                           eta=jetEta,\
                           phi=jetPhi))
    self.jets.sort(key=lambda jet: jet.pT, reverse=True)


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
    return self.jets[index]

  def __str__(self):
    return "Event object with %d Jet objects" % len(self.jets)
