# pill-matcher-python

This project contains Python code used either directly in the pill-matcher application, or in training/testing the
models used.  Each subdirectory addresses a different purpose, and will have different required libraries.

# PillServicesPythonEnv
This subdirectory contains the Python web services used by the [pill-matcher](https://github.com/lgviii/pill-project)
web application to generate inferences on an image file.  For the pill-matcher web application to work correctly, this 
application must also be run on the same system.

See [PillServicesPythonEnv README](PillServicesPythonEnv/README.md)

Code by Lin Greene, with color, shape, and OCR inference generation based on code by Mark Abrenio and Christina 
Molodowitch.

# color
This subdirectory contains scripts used in fine-tuning the color feature model used by the pill-matcher application.

It includes scripts for image download, resizing, and sorting, and model fine-tuning and inference generation.  It also 
includes a script for processing videos.

See [color README](color/README.md) for more information.

Code by Mark Abrenio.

# ocr
This subdirectory contains scripts used in testing the accuracy of OCR libraries in reading pill imprints.

See [ocr README](ocr/README.md) for more information.

Code by Christina Molodowitch.

# shape
This subdirectory contains scripts used in fine-tuning the shape feature model used by the pill-matcher application.

It includes scripts for image resizing and sorting, plus model fine-tuning and inference generation.

See [shape README](shape/README.md) for more information.

Code by Christina Molodowitch, based on code created by Mark for color model plus some early work on the shape model.
