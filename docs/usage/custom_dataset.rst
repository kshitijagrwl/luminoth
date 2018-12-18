Train Luminoth Faster-RCNN on Another Dataset
=============================================

This tutorial is to create a new dataset reader for your own dataset for
the `luminoth <https://github.com/tryolabs/luminoth>`__ faster-rcnn
model

I will illustrate how to train the Faster-RCNN on another dataset in the
following steps, and we will take **FLIR Night Driving Dataset** as the
example dataset.

Install luminoth
----------------

Install the luminoth package as mentioned in the README

.. code:: sh

    $ pip install luminoth

Build the train-set
-------------------

Get the Dataset
~~~~~~~~~~~~~~~

When you download and extract the `FILR
Dataset <https://www.flir.in/oem/adas/adas-dataset-form/>`__ you obtain
this architecture:

::

    /data/FLIR_ADAS
    ├── ADAS User License Agreement (26.Jul.2018) (Final).pdf
    ├── README - FLIR ADAS Dataset (31.Jul.2018) (Final).pdf
    ├── ReadMe.txt
    ├── training
    │   ├── AnnotatedPreviewData
    │   │   ├── FLIR_00001.jpeg
    │   │  
    │   ├── Annotations
    │   │   ├── FLIR_00001.json
    │   │
    │   ├── catids.json
    │   ├── Data
    │   │   ├── FLIR_00001.tiff
    │   │
    │   ├── PreviewData
    │   │   ├── FLIR_00001.jpeg
    │   │
    │   └── RGB
    │       ├── FLIR_00001.jpg
    ├── validation
    │   ├── AnnotatedPreviewData
    │   ├── Annotations
    │   ├── catids.json
    │   ├── Data
    │   ├── PreviewData
    │   └── RGB

We will use the PreviewData and Annotations folders from the training
and validation part of the dataset.

Add luminoth/tools/dataset/readers/object\_detection/yourdatabase.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the python site-packages folder you need to add the description of
your dataset reader in the file
``luminoth/tools/dataset/readers/object_detection/yourdatabase.py``

I will use the COCO and PASCAL readers as the baseline to create our
reader.

-  Define ``class FLIRReader(ObjectDetectionReader)`` with the
   constructor having details of the dataset
-  Define the ``iterate(self)`` function that yields the dataset to
   tfrecords
-  Define helper functions like ``get_classes`` or ``get_total``

.. code:: py

    def __init__(self, data_dir, split, **kwargs):
            super(FLIRReader, self).__init__(**kwargs)
            self._data_dir = data_dir
            self._split = split

            if split == "train":
                pathext = "training"
            elif split == "val":
                pathext = "validation"

            # self._labels_path = os.path.join(self._data_dir, 'ImageSets', 'Main')
            self._images_path = os.path.join(self._data_dir, pathext, 'PreviewData')
            self._annots_path = os.path.join(self._data_dir, pathext, 'Annotations')

            self.yielded_records = 0
            self.errors = 0


            categories = json.load(open(os.path.join(self._data_dir, pathext,'catids.json')))
            self._category_to_name = {
                c['id']: c['name'] for c in categories if c['id'] in [1,2,3,18,91]
            }
            self._total_classes = sorted(set(self._category_to_name.values()))

            # Validate PascalVoc structure in `data_dir`.
            self._validate_structure()

.. code:: py

    def iterate(self):
            # print self.classes
            for image_id in self._get_record_names():
                if self._stop_iteration():
                    # Finish iteration.
                    return

                if self._should_skip(image_id):
                    continue

                try:
                    annotation_path = self._get_image_annotation(image_id)
                    image_path = self._get_image_path(image_id)

                    # Read both the image and the annotation into memory.
                    annotation = json.load(tf.gfile.Open(annotation_path))
                    image = read_image(image_path)
                except tf.errors.NotFoundError:
                    tf.logging.debug(
                        'Error reading image or annotation for "{}".'.format(
                            image_id))
                    self.errors += 1
                    continue


                image_id = annotation['image']['file_name']
                gt_boxes = []
                
                for ann in annotation['annotation']:
                    x, y, width, height = ann['bbox']

                    # If the class is not in `classes`, it was filtered.
                    try:
            #           label_id = self.classes.index(b['name'])
                        annotation_class = self.classes.index(
                            self._category_to_name[int(ann['category_id'])]
                        )
                    except ValueError:
                        continue

                    gt_boxes.append({
                        'xmin': x,
                        'ymin': y,
                        'xmax': x + width,
                        'ymax': y + height,
                        'label': annotation_class,
                    })
                
                if len(gt_boxes) == 0:
                    continue

                record = {
                    'width': annotation['image']['width'],
                    'height': annotation['image']['height'],
                    'depth': 1,
                    'filename': annotation['image']['file_name'],
                    'image_raw': image,
                    'gt_boxes': gt_boxes,
                }
                self._will_add_record(record)
                self.yielded_records += 1

                yield record

Update init files
~~~~~~~~~~~~~~~~~

Then you need to link the new reader to relevant **init** files so that
it is discoverable. Add the following in
``luminoth/tools/dataset/readers/__init__.py``

.. code:: py

    READERS = {
              ...
              ...
              'flir':FLIRReader
              }

And in ``tools/dataset/readers/object_detection/__init__.py`` add:

.. code:: py

    from .flir import FLIRReader

Create the config file
----------------------

Create a config file to use for training the model. For example, if you
want to use the model **Resnet101** with alternate data inputs like the
FLIR dataset, you need to create a config file as follows:

.. code:: yaml

    train:
      # Run name for the training session.
      run_name: flir_da_block4
      job_dir: jobs
      batch_size: 4
      learning_rate:
        decay_method: piecewise_constant
        # Custom dataset for Luminoth Tutorial
        boundaries: [120000, 160000, 250000]
        values: [0.0003, 0.0001, 0.00003, 0.00001]
    dataset:
      type: object_detection
      #dir: /data/FLIR_ADAS
      dir: /data/anue_full
      data_augmentation:
          [{"flip":
              {"left_right": true,
                "prob": 0.5,
                "up_down": false}}]
    model:
      type: fasterrcnn
      batch_norm: true
      network:
        num_classes: 4
      anchors:
        # Add one more scale to be better at detecting small objects
        scales: [0.125, 0.25, 0.5, 1, 2]
        stride: 16
        ratios: [0.5, 1, 2]
      base_network:
        output_stride: 16
        architecture: resnet_v1_101
        train_batch_norm: false
        fine_tune_from: block4

Note - the number of classes in the FLIR dataset is 4

Convert the dataset
-------------------

Convert the dataset to tfrecords for input to the model

.. code:: sh

    $ lumi dataset transform --type flir --data-dir /data/FLIR_ADAS --split train --debug --output-dir /data/FLIR_ADAS

.. code:: sh

    $ lumi dataset transform --type flir --data-dir /data/FLIR_ADAS --split val --debug --output-dir /data/FLIR_ADAS

Launch the training
-------------------

Once the dataset is created, we can use it for training. Run the
following command in the shell.

.. code:: sh

    $ lumi train -c <path-to-config-file>.yml
