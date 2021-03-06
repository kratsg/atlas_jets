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
    x = np.abs(x)
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
class gGrid:
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
    uniform_towerdE = tower.Pt
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
    jet_energy = jet.Pt
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


# all coordinates are in (phi, eta) pairs always
#     and are converted to (cell_x, cell_y) at runtime
class OfflineGrid:
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


