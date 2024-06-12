from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compas.geometry.primitives import Frame
from compas.geometry.primitives import Primitive

# ellipse in compas 1 uses a plane in the constructor. Simple implementaiton with frame:


class EllipseFrame(Primitive):
    __slots__ = ["_frame", "_major", "_minor"]

    def __init__(self, frame, major, minor, **kwargs):
        super(EllipseFrame, self).__init__(**kwargs)
        self._frame = None
        self._major = None
        self._minor = None
        self.frame = frame
        self.major = major
        self.minor = minor

    @property
    def data(self):
        """dict : The data dictionary that represents the ellipse."""
        return {"frame": self.frame.data, "major": self.major, "minor": self.minor}

    @data.setter
    def data(self, data):
        self.frame = Frame.from_data(data["frame"])
        self.major = data["major"]
        self.minor = data["minor"]

    @classmethod
    def from_data(cls, data):
        return cls(Frame.from_data(data["frame"]), data["minor"], data["minor"])

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = Frame(*frame)

    @property
    def major(self):
        return self._major

    @major.setter
    def major(self, major):
        self._major = float(major)

    @property
    def minor(self):
        return self._minor

    @minor.setter
    def minor(self, minor):
        self._minor = float(minor)

    @property
    def normal(self):
        return self.frame.normal

    @property
    def center(self):
        return self.frame.point

    @center.setter
    def center(self, point):
        self.frame.point = point

    @property
    def area(self):
        raise NotImplementedError

    @property
    def circumference(self):
        raise NotImplementedError

    # ==========================================================================
    # customization
    # ==========================================================================

    def __repr__(self):
        return "EllipseFrame({0!r}, {1!r}, {2!r})".format(self.frame, self.major, self.minor)

    def __len__(self):
        return 3

    def __getitem__(self, key):
        if key == 0:
            return self.frame
        elif key == 1:
            return self.major
        elif key == 2:
            return self.minor
        else:
            raise KeyError

    def __setitem__(self, key, value):
        if key == 0:
            self.frame = value
        elif key == 1:
            self.major = value
        elif key == 2:
            self.minor = value
        else:
            raise KeyError

    def __iter__(self):
        return iter([self.frame, self.major, self.minor])

    # ==========================================================================
    # constructors
    # ==========================================================================

    # ==========================================================================
    # transformations
    # ==========================================================================

    def transform(self, T):
        self.frame.transform(T)
