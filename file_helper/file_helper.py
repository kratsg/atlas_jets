import os
import root_numpy as rnp

class RootFile:
  def __init__(self, filename = '', directory = '', tree = ''):
    self.filename  = filename
    self.directory = os.path.normpath(directory) #normalize path
    self.tree      = tree
    self.data      = ''
    self.load()

  def load(self):
    self.data = rnp.root2rec(self.filename, '%s/%s' % (self.directory, self.tree))
    print "Loaded %s:%s/%s" % (self.filename.split(os.sep)[-1], self.directory, self.tree)

  def __str__(self):
    return "ROOT data located at %s" % filename

