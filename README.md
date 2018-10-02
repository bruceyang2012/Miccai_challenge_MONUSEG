## 7th place on Miccai - Multi Organ Nuclei Segementation
Chanllenge results: https://monuseg.grand-challenge.org/Results/

Slides: https://docs.google.com/presentation/d/1jS9YEs_KVBamoYdEZ0oSGUbIBQmr2htOz12dQLdf4Sk/edit?usp=sharing

Manuscript: https://drive.google.com/open?id=1S1apR4SV_aCiFbfLCaAkhh3EpJCfDCDu

---

Please install package below
```python
pip install numba numexpr pygsheets oauth2client
```

First, setup your model hyper-parameter config in the **monuconfig.py**. We support backone: resnet50/101, densenet121/169 and inception-resnetv2, please specify model in BACKBONE.
```python
class Config(object):
  NAME = "name your model"
  RPN_ANCHOR_SCALE = (2, 4, 6, 8, 10)
  BACKBONE = "resnet101"
  ...
```

Now support Path Aggregation Network and used as default. If you want to use original Mask-RCNN, please revise code in **train.py** when creating model

```python
  # Create model
  model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs, is_PANet=False)
```



Then train the Mask-RCNN by
```
python train.py --weight imagenet --dataset dataset/ --logs logs/ --subset train
```

-------
Already implemented features
- Path Aggregation Network
- Speed up data generator by 
  - feed all data into memory first
  - apply Numba on utils.compute_overlap
  - rewrite utils.extract_boxex
  - revise some indexing code
- Support more pre-train model structure like DenseNet, Inception-Resnetv2
- Config and AJI results will be automatically recored on [gsheets](https://docs.google.com/spreadsheets/d/1bsn77IhLudcricP9VXeIycU-S5lByRo3mKUzlYjeFow/edit#gid=0)
- Speed up AJI code (implemented by 旻昇, 友誠)

TODO
- **Synchronize Batch Normalization**
- soft-NMS
- relation network
- Attetion on FPN

