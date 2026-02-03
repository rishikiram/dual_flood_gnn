## Data
code base is desinged to process shapefile, DEM, and maybe other files to produce a 'database' of static/dynamic node/edge features. This has already been done for us and posted on the Kaggle.

Also, this model is desinged to account for total conservation of water, physics-informed style. For this, the boundary conditions must be defined. The Kaggle does not provide boundary conditions, so my ideas for dealing with this are: turnoff conservation of water loss, or two learn boundary conditions.