#!/usr/bin/env python
from ROOT import TLorentzVector
import cPickle as pickle


class MyVec(TLorentzVector):
    def __init__(self, *args, **kwargs):
        print "init"
        if len(args) == 2 and isinstance(args[0], pow.__class__):
          super(self.__class__, self).__init__(args[0](*args[1]))
        else:
          args = args or (TLorentzVector())
          if isinstance(args[0], TLorentzVector):
            super(self.__class__, self).__init__(args[0])
          else:
            raise ValueError("Unexpected value")
        self.a = kwargs.get('a', 'fake')
        self.b = kwargs.get('b', TLorentzVector())
        print "args", args
        print "kwargs", kwargs

    def __setstate__(self, state):
        print "setstate"
        self.__dict__ = state

    def __getstate__(self):
        print "getstate"
        return self.__dict__

    def __reduce__(self):
        print "reduce"
        return (self.__class__, super(self.__class__, self).__reduce__(), self.__getstate__(), )

anotherVec = TLorentzVector()
anotherVec.SetPtEtaPhiM(50.0, 0.0, 0.0, 0.0)

evenMore = TLorentzVector()
evenMore.SetPtEtaPhiM(100.0, 0.0, 0.0, 0.0)

a = MyVec(anotherVec, a='testing', b=evenMore)

b = pickle.loads(pickle.dumps(a))

print a.__class__
print b.__class__

print a.__dict__
print b.__dict__

print a.Pt()
print b.Pt()

print a.b.Pt()
print b.b.Pt()

print a == b
print isinstance(a, MyVec)
print isinstance(b, MyVec)
