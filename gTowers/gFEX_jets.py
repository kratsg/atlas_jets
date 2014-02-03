''' Class definitions for dealing with ATLAS jets using towers'''

#ROOT is needed to deal with rootfiles
import ROOT
#import TLorentzVector to do a vector-sum
from ROOT import TLorentzVector

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
               recon_algo            = 'uniform'):
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
    self.towers = []
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

  def add_tower_event(self, tower_event):
    for tower in tower_event:
      self.add_tower(tower)

  def add_event(self, event):
    for jet in event:
      self.add_jet(jet)

  def add_tower(self, tower):
    '''add a single `tower` to the current grid'''
    #grab the boundaries of the rectangular region adding
    minX, minY = self.phieta2cell((tower.phiMin, tower.etaMin))
    maxX, maxY = self.phieta2cell((tower.phiMax, tower.etaMax))

    #build up the rectangular grid for the coordinates
    tower_mesh_coords = self.__rectangular_mesh( minX, maxX, minY, maxY )
    uniform_towerdE = tower.pT
    tower_mesh = ([tuple(cell_coord), uniform_towerdE] for cell_coord in tower_mesh_coords if self.boundary_conditions(cell_coord) )
    for cell_coord, fractional_energy in tower_mesh:
      try:
        if cell_coord[0] >= self.grid.shape[0]:
          cell_coord = (cell_coord[0] - self.grid.shape[0], cell_coord[1])
        self.grid[cell_coord] += fractional_energy
      except IndexError:
        print "\t"*2, tower
        print "\t"*2, '-- tower cell_coord could not be added:', cell_coord

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
    # fix scaling for plots
    normalization = 2. * np.pi * erf( 0.92 * (2.**-0.5) )**2.#* cell_radius**2.
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
      uniform_jetdPt = jet_energy#/(square_mesh_coords.size/2.)
      mesh = ([tuple(cell_coord), uniform_jetdPt] for cell_coord in square_mesh_coords if self.boundary_conditions(cell_coord) )
    elif self.recon_algo == 'gaussian':
      mesh = ([tuple(cell_coord), self.__jetdPt(cell_jetcoord, cell_radius, jet_energy, cell_coord)] for cell_coord in square_mesh_coords if self.boundary_conditions(cell_coord)&circularRegion(cell_jetcoord, cell_radius, cell_coord) )
    return mesh

  def __rectangular_mesh(self, minX, maxX, minY, maxY):
    i,j = np.meshgrid( np.arange( minX, maxX), np.arange( minY, maxY) )
    return np.transpose([i.reshape(-1), j.reshape(-1)]).astype(int)

  def __square_mesh(self, center, radius):
    '''generates a square meshgrid of points for center and sidelength 2*radius'''
    return self.__rectangular_mesh(center[0] - radius, center[0] + radius + 1, center[1] - radius, center[1] + radius + 1)

  def __make_plot(self, title='Grid Plot', colzLabel = '$E_T$'):
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
    # set the colorbar
    cbar = pl.colorbar(pad=0.2)
    cbar.set_label('%s [GeV]' % colzLabel)
    return fig

  def show(self, title='Grid Plot', colzLabel = '$E_T$'):
    '''Show an image of the current grid'''
    fig = self.__make_plot(title, colzLabel)
    fig.show()
    pl.close(fig)

  def save(self, title='Grid Plot', filename='output.png', colzLabel = '$E_T$'):
    '''Save an image of the current grid to file'''
    fig = self.__make_plot(title, colzLabel)
    fig.savefig(filename)
    pl.close(fig)

  def __str__(self):
    return "Grid object:\n\tPhi: %s\n\tEta: %s\n\tResolution: %0.2f" % (self.domain[0], self.domain[1], self.cell_resolution)

class SeedFilter:
  def __init__(self, ETthresh = 0, numSeeds = 1.0e5):
    self.ETthresh = ETthresh
    self.numSeeds = int(numSeeds)

  def filter(self, seeds):
    return [seed for seed in seeds if seed.E > self.ETthresh][:self.numSeeds]

  def __call__(self, seeds):
    return self.filter(seeds)

  def __str__(self):
    return "SeedFilter object returning at most %d seeds > %0.4f GeV" % (self.numSeeds, self.ETthresh)

class Jet:
  def __init__(self,\
               inputThresh    = 200.,\
               triggerThresh  = 150.,\
               eta            = 0.0,\
               phi            = 0.0,\
               TLorentzVector = TLorentzVector(),\
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
    self.phi            = np.float(phi)
    self.eta            = np.float(eta)
    self.coord          = (self.phi, self.eta)
    self.TLorentzVector = TLorentzVector
    # setting up basic details from vector
    self.E = self.TLorentzVector.E()
    self.pT = self.TLorentzVector.Pt()
    self.m = self.TLorentzVector.M()
    # setting up jet details
    self.radius         = np.float(radius)
    self.input_energy   = np.float(self.pT)
    self.trigger_energy = np.float(self.pT)

  def __str__(self):
    return "Jet object:\n\tPhi: %0.4f\n\tEta: %0.4f\n\tE: %0.2f (GeV)\n\tpT: %0.2f (GeV)\n\tm: %0.2f (GeV)\n\tInputted: %s\n\tTriggered: %s" % (self.phi, self.eta, self.E, self.pT, self.m, self.inputted(), self.triggered())

  def inputted(self):
    return self.input_energy > self.inputThresh

  def triggered(self):
    return self.trigger_energy > self.triggerThresh

# to be grammatically correct, it should be Events' Towers
class TowerEvents:
  def __init__(self, rootfile, seed_filter = SeedFilter()):
    self.rootfile    = rootfile
    self.events      = []
    self.seed_filter = seed_filter
    self.load()

  def load(self):
    indices = [self.rootfile.data.dtype.names.index(name) for name in self.rootfile.data.dtype.names if 'gTower' in name]
    self.events = [TowerEvent(event=[event[i] for i in indices], seed_filter=self.seed_filter) for event in self.rootfile.data]
    print 'Loaded gTower jets'

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
    return "TowerEvents object with %d TowerEvent objects" % len(self.events)

class TowerEvent:
  def __init__(self, event = [], seed_filter = SeedFilter()):
    self.towers = []
    self.seed_filter = seed_filter
    # note that unlike David's data, it isn't a "tuple" of 215 items
    # holy mother of god, please do not blame me for the fact that
    #    I'm ignoring like 210 items in this list, we only want gTower info
    for gTowerE, gTowerNCells, gTowerEtaMin, gTowerEtaMax, gTowerPhiMin, gTowerPhiMax in zip(*event):
      self.towers.append(Tower(E=gTowerE,\
                               num_cells=gTowerNCells,\
                               etaMin=gTowerEtaMin,\
                               etaMax=gTowerEtaMax,\
                               phiMin=gTowerPhiMin,\
                               phiMax=gTowerPhiMax))

    self.E = [tower.E for tower in self.towers]
    self.phiMin = np.min([tower.phiMin for tower in self.towers])
    self.etaMin = np.min([tower.etaMin for tower in self.towers])
    self.phiMax = np.max([tower.phiMax for tower in self.towers])
    self.etaMax = np.max([tower.etaMax for tower in self.towers])

  def set_seed_filter(self, seed_filter):
    self.seed_filter = seed_filter

  def filter_towers(self):
    if not self.seed_filter:
      raise ValueError("You must set a filter with self.seed_filter(*SeedFilter).")
    return self.seed_filter(self.towers)

  def get_event(self):
    self.__seeds_to_jet()
    return self.event

  def __seeds_to_jet(self):
    jets = []
    for seed in self.filter_towers():
      jets.append(self.__seed_to_jet(seed))
    self.event = Event(jets=jets)

  def __seed_to_jet(self, seed):
    # note: each tower has m=0, so E = p, ET = pT
    l = seed.TLorentzVector
    for tower in self.__towers_around(seed):
      radius = 1.0
      normalization = 2. * np.pi * radius**2. * erf( 0.92 * (2.**-0.5) )**2.
      exponential = np.exp(-( (seed.phi - tower.phi)**2./(2. * (radius**2.)) + (seed.eta - tower.eta)**2./(2.*(radius**2.)) ))
      towerTLorentzVector = TLorentzVector()
      towerTLorentzVector.SetPtEtaPhiM(tower.E/np.cosh(tower.eta) * normalization/exponential, tower.eta, tower.phi, 0.0)
      l += towerTLorentzVector
    return Jet(eta=seed.eta, phi=seed.phi, TLorentzVector = l)

  def __towers_around(self, seed, radius=1.0):
    return [tower for tower in self.towers if np.sqrt((tower.phi - seed.phi)**2. + (tower.eta - seed.eta)**2.) <= radius]

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
    return "TowerEvent object with %d Tower objects\n\tphi: (%0.4f, %0.4f)\n\teta: (%0.4f, %0.4f)" % (len(self.towers), self.phiMin, self.phiMax, self.etaMin, self.etaMax)

# to do -- fill this shit in
class Tower:
  def __init__(self,\
               E,\
               num_cells,\
               etaMin,\
               etaMax,\
               phiMin,\
               phiMax):
    self.E = E
    self.num_cells = num_cells
    self.etaMin = etaMin
    self.etaMax = etaMax
    self.phiMin = phiMin
    self.phiMax = phiMax
    # set the center of the tower to the geometric center
    self.eta = (self.etaMax + self.etaMin)/2.0
    self.phi = (self.phiMax + self.phiMin)/2.0
    # calculate pT
    self.pT = self.E/np.cosh(self.eta)
    # generate a TLorentzVector to handle additions
    #   note: m = 0 for towers, so E = p --> ET = pT
    self.TLorentzVector = TLorentzVector()
    self.TLorentzVector.SetPtEtaPhiM(self.pT, self.eta, self.phi, 0.0)

  def __str__(self):
    return "Tower object:\n\tE: %0.4f (GeV)\n\tnum_cells: %d\n\tphi: (%0.4f,%0.4f) \td = %0.4f\n\teta: (%0.4f, %0.4f) \td = %0.4f" % (self.E, self.num_cells, self.phiMin, self.phiMax, self.phiMax - self.phiMin, self.etaMin, self.etaMax, self.etaMax - self.etaMin)

class Events:
  def __init__(self, events = []):
    self.events = events

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
  def __init__(self, jets = []):
    self.jets = jets
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

class Analysis:
  def __init__(self, events = [], num_bins = 50):
    self.events = events
    self.num_bins = num_bins

  def Efficiency(self):
    input_jets = np.array([jet.pT for event in self.events for jet in event[:2]])
    trigger_jets = np.array([jet.pT for event in self.events for jet in event[:2] if jet.triggered()])

    bin_range = (input_jets.min(), input_jets.max())
    histogram_input = np.histogram(input_jets, range=bin_range, bins=self.num_bins)
    histogram_trigger = np.histogram(trigger_jets, range=bin_range, bins=self.num_bins)
    nonzero_bins = np.where(histogram_input[0] != 0)
    efficiency = np.true_divide(histogram_trigger[0][nonzero_bins], histogram_input[0][nonzero_bins])

    pl.figure()
    pl.scatter(histogram_input[1][nonzero_bins], efficiency)
    pl.xlabel('$\mathrm{p}_{\mathrm{T}}^{\mathrm{jet}}$ [GeV]')
    pl.ylabel('Efficiency')
    pl.title('Turn-on Plot')
    pl.show()
