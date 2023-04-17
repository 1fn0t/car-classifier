The dependencies of the project are in the requirements.txt file. Preperably use a separate anaconda environment for this project.
Particularly numpy create issues as otehr dependencies depend on it. The project is set up to work with a gpu but the first 2
lines of main can be deleted or commendted out to use CPU.

The dataset this project uses is a tensorflow dataset so upon running the main.py file for the first time it will download the dataset.
