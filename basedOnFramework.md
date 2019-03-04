## input
``` cmd
python bm_dataset/create_real_synth_dataset.py -i ./data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.jpg -l ./data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.csv -o ./output/synth_dataset -n 5 --nb_workers 3 --visual
```

# cmd output
``` cmd
No display found. Using non-interactive Agg backend
INFO:root:ARGUMENTS:
 {'path_image': './data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.jpg', 'path_landmarks': './data_images/rat-kidney_/scale-5pc/Rat_Kidney_HE.csv', 'path_out': './output/synth_dataset', 'nb_samples': 5, 'visual': True, 'nb_workers': 3}
INFO:root:running...
WARNING:root:using existing folder: D:\GitHub\BIRL\output\synth_dataset
  0%|                                                                                            | 0/5 [00:00<?, ?it/s]No display found. Using non-interactive Agg backend
No display found. Using non-interactive Agg backend
No display found. Using non-interactive Agg backend
multiprocessing.pool.RemoteTraceback:
"""
Traceback (most recent call last):
  File "D:\Anaconda3\envs\tensorflow\lib\multiprocessing\pool.py", line 119, in worker
    result = (True, func(*args, **kwds))
  File "D:\GitHub\BIRL\bm_dataset\create_real_synth_dataset.py", line 274, in perform_deform_export
    max_deform)
  File "D:\GitHub\BIRL\bm_dataset\create_real_synth_dataset.py", line 170, in deform_image_landmarks
    method='linear', fill_value=1.)
  File "D:\Anaconda3\envs\tensorflow\lib\site-packages\scipy\interpolate\ndgriddata.py", line 222, in griddata
    rescale=rescale)
  File "interpnd.pyx", line 245, in scipy.interpolate.interpnd.LinearNDInterpolator.__init__
  File "interpnd.pyx", line 80, in scipy.interpolate.interpnd.NDInterpolatorBase.__init__
  File "interpnd.pyx", line 190, in scipy.interpolate.interpnd._check_init_shape
IndexError: tuple index out of range
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "bm_dataset/create_real_synth_dataset.py", line 339, in <module>
    main(arg_params)
  File "bm_dataset/create_real_synth_dataset.py", line 323, in main
    range(params['nb_samples'])):
  File "D:\Anaconda3\envs\tensorflow\lib\multiprocessing\pool.py", line 735, in next
    raise value
IndexError: tuple index out of range

```

## input
``` cmd
python bm_dataset/generate_regist_pairs.py -i ./output/synth_dataset/*.jpg -l ./output/synth_dataset/*.csv -csv ./output/cover_synth-dataset.csv --mode each2all
```
