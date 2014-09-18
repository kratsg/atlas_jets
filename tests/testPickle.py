'''
http://stackoverflow.com/questions/25733977/how-to-pickle-classes-which-inherit-from-tlorentzvector/25736113#25736113
http://stackoverflow.com/questions/8574742/how-to-pickle-an-object-of-a-class-b-having-many-variables-that-inherits-from
http://stackoverflow.com/questions/3554310/how-to-best-pickle-unpickle-in-class-hierarchies-if-parent-and-child-class-insta?rq=1
http://stackoverflow.com/questions/19855156/whats-the-exact-usage-of-reduce-getstate-setstate-in-pickler
http://docs.aakashlabs.org//apl/pyhelp/pydocs/library/pickle.html#pickling-and-unpickling-extension-types
'''

from atlas_jets import gTowers
import cPickle as pickle

null_gTower = gTowers.Tower(et=100.0, etamin=0.0, etamax=1.0, phimin=1.0, phimax=3.0, num_cells=23)
test = gTowers.Jet(null_gTower, area=3.14159, radius=1.0, towers=[null_gTower, null_gTower, null_gTower], seed=null_gTower)
test2 = pickle.loads(pickle.dumps(test))
print test.__class__
print test2.__class__

print test.__dict__
print test2.__dict__

print test == test2
print test.seed
print test2.seed

print test.towers[1]
print test2.towers[1]
