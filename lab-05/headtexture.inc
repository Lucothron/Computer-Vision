// Define the texture of the face and specify the lighting and background

#include "textures.inc"
#include "colors.inc" 
#include "finish.inc"

global_settings { assumed_gamma 2.2 }

// Texture declaration for object "FACE"

#declare FACE = texture {
   pigment { color rgb <1, 0.7, 0.65> } 
   finish { 
     ambient 0.2 
     diffuse 0.7 
   }   
}

// Texture declareation for object "NOSE"

#declare NOSE = texture {
   pigment { color rgb <1, 0.7, 0.65> } 
   finish { 
     ambient 0.27 
     diffuse .95
   }
}

light_source {
  <200, 1000, 1000> White
  shadowless
}

sky_sphere {
  pigment { rgb <0.5, 0.5, 0.75> }
}
