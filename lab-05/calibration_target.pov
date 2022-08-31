//-----------------------------------------------------------------------------
// C A L I B R A T I O N   T A R G E T  for POV-ray
//-----------------------------------------------------------------------------
// REVISION HISTORY
//  0.00 2011-09-04 Adrian F Clark <alien@essex.ac.uk>
//       Original version
//-----------------------------------------------------------------------------

#include "colors.inc"

global_settings {
  assumed_gamma 1.0
}

light_source {
  <200, 1000, 1000> White
  shadowless
}

plane { z, 270 texture { pigment { Green } } }

box { <  0, 300, 280>, < 20, 320, 260> texture { pigment { color Black } } }
box { < 20, 300, 280>, < 40, 320, 260> texture { pigment { color White } } }
box { < 40, 300, 280>, < 60, 320, 260> texture { pigment { color Black } } }
box { < 60, 300, 280>, < 80, 320, 260> texture { pigment { color White } } }
box { < 80, 300, 280>, <100, 320, 260> texture { pigment { color Black } } }
box { <100, 300, 280>, <120, 320, 260> texture { pigment { color White } } }
box { <120, 300, 280>, <140, 320, 260> texture { pigment { color Black } } }
box { <140, 300, 280>, <160, 320, 260> texture { pigment { color White } } }

box { <  0, 280, 280>, < 20, 300, 260> texture { pigment { color White } } }
box { < 20, 280, 280>, < 40, 300, 260> texture { pigment { color Black } } }
box { < 40, 280, 280>, < 60, 300, 260> texture { pigment { color White } } }
box { < 60, 280, 280>, < 80, 300, 260> texture { pigment { color Black } } }
box { < 80, 280, 280>, <100, 300, 260> texture { pigment { color White } } }
box { <100, 280, 280>, <120, 300, 260> texture { pigment { color Black } } }
box { <120, 280, 280>, <140, 300, 260> texture { pigment { color White } } }
box { <140, 280, 280>, <160, 300, 260> texture { pigment { color Black } } }

box { <  0, 260, 280>, < 20, 280, 260> texture { pigment { color Black } } }
box { < 20, 260, 280>, < 40, 280, 260> texture { pigment { color White } } }
box { < 40, 260, 280>, < 60, 280, 260> texture { pigment { color Black } } }
box { < 60, 260, 280>, < 80, 280, 260> texture { pigment { color White } } }
box { < 80, 260, 280>, <100, 280, 260> texture { pigment { color Black } } }
box { <100, 260, 280>, <120, 280, 260> texture { pigment { color White } } }
box { <120, 260, 280>, <140, 280, 260> texture { pigment { color Black } } }
box { <140, 260, 280>, <160, 280, 260> texture { pigment { color White } } }

box { <  0, 240, 280>, < 20, 260, 260> texture { pigment { color White } } }
box { < 20, 240, 280>, < 40, 260, 260> texture { pigment { color Black } } }
box { < 40, 240, 280>, < 60, 260, 260> texture { pigment { color White } } }
box { < 60, 240, 280>, < 80, 260, 260> texture { pigment { color Black } } }
box { < 80, 240, 280>, <100, 260, 260> texture { pigment { color White } } }
box { <100, 240, 280>, <120, 260, 260> texture { pigment { color Black } } }
box { <120, 240, 280>, <140, 260, 260> texture { pigment { color White } } }
box { <140, 240, 280>, <160, 260, 260> texture { pigment { color Black } } }

box { <  0, 220, 280>, < 20, 240, 260> texture { pigment { color Black } } }
box { < 20, 220, 280>, < 40, 240, 260> texture { pigment { color White } } }
box { < 40, 220, 280>, < 60, 240, 260> texture { pigment { color Black } } }
box { < 60, 220, 280>, < 80, 240, 260> texture { pigment { color White } } }
box { < 80, 220, 280>, <100, 240, 260> texture { pigment { color Black } } }
box { <100, 220, 280>, <120, 240, 260> texture { pigment { color White } } }
box { <120, 220, 280>, <140, 240, 260> texture { pigment { color Black } } }
box { <140, 220, 280>, <160, 240, 260> texture { pigment { color White } } }

box { <  0, 200, 280>, < 20, 220, 260> texture { pigment { color White } } }
box { < 20, 200, 280>, < 40, 220, 260> texture { pigment { color Black } } }
box { < 40, 200, 280>, < 60, 220, 260> texture { pigment { color White } } }
box { < 60, 200, 280>, < 80, 220, 260> texture { pigment { color Black } } }
box { < 80, 200, 280>, <100, 220, 260> texture { pigment { color White } } }
box { <100, 200, 280>, <120, 220, 260> texture { pigment { color Black } } }
box { <120, 200, 280>, <140, 220, 260> texture { pigment { color White } } }
box { <140, 200, 280>, <160, 220, 260> texture { pigment { color Black } } }

box { <  0, 180, 280>, < 20, 200, 260> texture { pigment { color Black } } }
box { < 20, 180, 280>, < 40, 200, 260> texture { pigment { color White } } }
box { < 40, 180, 280>, < 60, 200, 260> texture { pigment { color Black } } }
box { < 60, 180, 280>, < 80, 200, 260> texture { pigment { color White } } }
box { < 80, 180, 280>, <100, 200, 260> texture { pigment { color Black } } }
box { <100, 180, 280>, <120, 200, 260> texture { pigment { color White } } }
box { <120, 180, 280>, <140, 200, 260> texture { pigment { color Black } } }
box { <140, 180, 280>, <160, 200, 260> texture { pigment { color White } } }

box { <  0, 160, 280>, < 20, 180, 260> texture { pigment { color White } } }
box { < 20, 160, 280>, < 40, 180, 260> texture { pigment { color Black } } }
box { < 40, 160, 280>, < 60, 180, 260> texture { pigment { color White } } }
box { < 60, 160, 280>, < 80, 180, 260> texture { pigment { color Black } } }
box { < 80, 160, 280>, <100, 180, 260> texture { pigment { color White } } }
box { <100, 160, 280>, <120, 180, 260> texture { pigment { color Black } } }
box { <120, 160, 280>, <140, 180, 260> texture { pigment { color White } } }
box { <140, 160, 280>, <160, 180, 260> texture { pigment { color Black } } }

box { <  0, 140, 280>, < 20, 160, 260> texture { pigment { color Black } } }
box { < 20, 140, 280>, < 40, 160, 260> texture { pigment { color White } } }
box { < 40, 140, 280>, < 60, 160, 260> texture { pigment { color Black } } }
box { < 60, 140, 280>, < 80, 160, 260> texture { pigment { color White } } }
box { < 80, 140, 280>, <100, 160, 260> texture { pigment { color Black } } }
box { <100, 140, 280>, <120, 160, 260> texture { pigment { color White } } }
box { <120, 140, 280>, <140, 160, 260> texture { pigment { color Black } } }
box { <140, 140, 280>, <160, 160, 260> texture { pigment { color White } } }

