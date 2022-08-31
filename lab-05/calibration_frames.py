#!/usr/bin/env python3
'''Generate a series of calibration frames using POV-ray.'''
from __future__ import division
import sys, os, math

def do_scene (x, y, z, fn):
    '''Generate a frame with the camera at x,y,z into fn and render it.'''
    f = open (fn, 'w')
    print ('#include "calibration_target.pov"', file=f)
    print ('camera {', file=f)
    print ('  location <%.2f, %.2f, %.2f>' % (x, y, z), file=f)
    print ('  look_at <%.2f, 300, 280>' % x, file=f)
    print ('}', file=f)
    f.close ()
    os.system ('povray +I%s +FN +W640 +H480 +AA +A0.3 -D &> /dev/null' % fn)

# Main program: calculate the camera positions and generate the frames.
n = 30
for i in range (0, n):
    x =  75 + 100 * math.cos (i * math.pi / n)
    y =  50 + 100 * math.cos (i * math.pi / n)
    z = 650 + 100 * math.sin (i * math.pi / n)
    print (y, z)
    fn = 'calib-%3.3d.pov' % i
    do_scene (x, y, z, fn)
