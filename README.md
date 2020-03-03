# Astronomy-Tracking

This is the first part of my project for Computer Science 600. This module will allow you to automatically generate a lightcurve for
a set of astronomical images. Below are the instructions to use it.

First, install [Python](https://www.python.org/downloads/).

Next, install or upgrade [Pip](https://pip.pypa.io/en/stable/installing/).

Open command prompt (also known as terminal or command line, depending on what operating system you have). Go to the folder where
you downloaded the project and run this command:

`pip install -r requirements.txt`

This should install all of the packages that you need to run this program. Once you do this, you will be able to run this application
with this command:

`flask run`

Once you have run this, open your internet browser and go to [localhost:5000](localhost:5000). This does not require an internet connection as all of the code
will be running on your computer.

I have included an example astronomy dataset which can be found in Data/Raw. Running the flask application will denoise these astronomical
images, allow you to choose a star to track throughout the images, and create a light curve. You can see the images after denoising in Data/Testing/Denoised
The positions of the star in each image are stored as positions.npy and the lightcurve is saved as Data/lightcurve.png

If you want to run this program with your own set of images, delete the images, positions, and .fits files currently in the Data folder and put your .fits files into Data/Raw.

This is still a work in progress, so some errors or unoptimized portions still exist.
