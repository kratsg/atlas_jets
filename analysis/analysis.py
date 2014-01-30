# numpy and matplotlib (mpl) are used for computing and plotting 
import numpy as np
import matplotlib.pyplot as pl

class Analysis:
  def __init__(self, offline_events = [], tower_events = []):
    self.offline_events = offline_events
    self.tower_events = tower_events

  def Efficiency(self, num_bins = 50):
    print "Needs to be worked on"
    return False
    input_jets = np.array([jet.pT for event in self.offline_events for jet in event[:2]])
    trigger_jets = np.array([jet.pT for event in self.offline_events for jet in event[:2] if jet.triggered()])

    bin_range = (input_jets.min(), input_jets.max())
    histogram_input = np.histogram(input_jets, range=bin_range, bins=num_bins)
    histogram_trigger = np.histogram(trigger_jets, range=bin_range, bins=num_bins)
    nonzero_bins = np.where(histogram_input[0] != 0)
    efficiency = np.true_divide(histogram_trigger[0][nonzero_bins], histogram_input[0][nonzero_bins])

    pl.figure()
    pl.scatter(histogram_input[1][nonzero_bins], efficiency)
    pl.xlabel('$\mathrm{p}_{\mathrm{T}}^{\mathrm{jet}}$ [GeV]')
    pl.ylabel('Efficiency')
    pl.title('Turn-on Plot')
    pl.show()

  def TowerMultiplicity(self, pT_thresh = 200.):
    i = 0
    ETmax = np.max([tower.E/np.cosh(tower.eta) for tower_event in self.tower_events for tower in tower_event.towers])
    bin_edges = np.arange(0,ETmax + 5., 5.)
    num_bins = len(bin_edges)-1
    cumul_sum = np.zeros(num_bins).astype(int)
    #number of events that pass threshold - for scaling to get multiplicity
    num_events = 0

    for tower_event, offline_event in zip(self.tower_events, self.offline_events):
      try:
        if len(offline_event.jets) == 0 or offline_event.jets[0].pT < pT_thresh:
          continue
      except IndexError:
        continue
      num_events += 1
      ETs = [tower.E/np.cosh(tower.eta) for tower in tower_event.towers]
      #now we want the histogram of towerThresholds, so we need to count # towers above a certain threshold of ET
      hist, _ = np.histogram(ETs, bins=bin_edges)
      cumul_sum += np.cumsum(hist[::-1])[::-1]

    print "%d offline events after the cut" % num_events
    #get width of each bar based on domain/num_bins
    width=[x - bin_edges[i-1] for i,x in enumerate(bin_edges)][1:]
    #normalize distribution for multiplicity
    cumul_sum = 1.0*cumul_sum/num_events
    # plot it all
    pl.figure()
    pl.xlabel('$E_T^{\mathrm{threshold}}$ [GeV]')
    pl.ylabel('Number of gTowers')
    pl.title('Number of gTowers above $E_T^{\mathrm{threshold}}$ for $p_T^{\mathrm{jet}}$ > %d GeV' % pT_thresh)
    pl.bar(bin_edges[:-1], cumul_sum, width=width, log=True)
    #pl.xscale('log')
    #pl.yscale - need to use log=True argument in pyplot.bar (see documentation)
    pl.savefig('events_threshold_histogram_multiplicity%d.png' % pT_thresh)
    pl.close()

  def TowerHistogram(self, pT_thresh = 200.):
    i = 0
    bin_edges = np.array([0,50,100,150,200,250,300,350,400,500,750,1000,4000]).astype(float)
    num_bins = len(bin_edges)-1
    cumul_sum = np.zeros(num_bins).astype(int)
    #number of events that pass threshold - for scaling to get multiplicity
    num_events = 0

    for tower_event, offline_event in zip(self.tower_events, self.offline_events):
      try:
        if len(offline_event.jets) == 0 or offline_event.jets[0].pT < pT_thresh:
          continue
      except IndexError:
        continue
      num_events += 1
      ETs = [tower.E/np.cosh(tower.eta) for tower in tower_event.towers]
      #now we want the histogram of towerThresholds, so we need to count # towers above a certain threshold of ET
      hist, _ = np.histogram(ETs, bins=bin_edges)
      cumul_sum += hist

    print "%d offline events after the cut" % num_events
    #get width of each bar based on domain/num_bins
    width=[x - bin_edges[i-1] for i,x in enumerate(bin_edges)][1:]
    #normalize distribution for multiplicity
    cumul_sum = 1.0*cumul_sum/num_events
    # plot it all
    pl.figure()
    pl.xlabel('$E_T^{\mathrm{threshold}}$ [GeV]')
    pl.ylabel('Number of gTowers')
    pl.title('Histogram of gTowers for $p_T^{\mathrm{jet}}$ > %d GeV' % pT_thresh)
    pl.bar(bin_edges[:-1], cumul_sum, width=width, log=True)
    #pl.xscale('log')
    #pl.yscale - need to use log=True argument in pyplot.bar (see documentation)
    pl.savefig('events_threshold_histogram_towers%d.png' % pT_thresh)
    pl.close()


