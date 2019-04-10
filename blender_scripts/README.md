Scripts for rendering stuff in Blender.

Every script renders all models placed in *models* into *render*. 
Mind placing the models in the correct format, i.e. *.obj* for meshes and *.ply* for point clouds.
The scripts must be executed from within Blender or as command line parameter, e.g.
```
blender -b -P randview_color.py
```

Models are expected to be bound within (-0.5 .. 0.5)(-0.5 .. 0.5)(0 .. 1) to be correctly visible.