# :running_woman: Inference Code readme

### do novel view synthesis given 2D images (on some demo images):
We render `azim` novel view video by default.
```bash
bash scripts/test/demo_view_synthesis.sh
```

### Conduct semantics editing (on some demo images):
```bash
bash scripts/test/demo_editing.sh
```

This script shall output a video with the change of yaw angle and editid scale. To edit a specific attribute on an identity, the following flags need to be set:
```--smile_ids```, ```--beard_ids```, ```--bangs_ids``` and ```--age_ids```.

To control the editing scale, set the following flags:

```---editing_boundary_scale_upperbound``` and ```---editing_boundary_scale_lowerbound```, following the order of Bangs, Smiling, No_Beard and Young. We also have the editing directions of Eyeglass, but this seems to be unstable on StyleSDF and we do not include the results in the final paper.


### 3D Toonifications with our pre-triaind encoder:
Here we use a fine-tuned 3D GAN with pre-trained `E0` encoder (w/o local feature module) to do the inference.
```bash
bash scripts/test/demo_toonify.sh
```

### Reproduce the results in Table 1 (Quantitative performance on CelebA-HQ.)

```bash
bash scripts/test/eval_2dmetrics_ffhq.sh
```

------

### Render video flags
change the following code to determin some characteristics of the video rendered.

```
--render_video \ # whether to render video,
--azim_video \ # render azim/ellipsoid video
--video_frames 9 \ # how many frames in the vide 
--no_surface_renderings \ # whether to render the mesh
```

### Misc
Note that for the scripts end with `_ada`, we use the stage 2 (pure 2D alignment) model to do the inference. It usually yields results with better fidelity but less view consistency.  Feel free to try and use the model that suits your need. To download this model, first run `python download_ada_models.py` to download the pre-trained encoder.