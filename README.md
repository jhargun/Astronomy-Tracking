# Astronomy-Tracking

This is the first part of my project for Computer Science 600. This module will allow you to track stars through multiple astronomical
images. To use it:

Make sure you have numpy, opencv, astropy, flask, and flask-wtf installed. Then, run this command:

`flask run`

I have included an example astronomy dataset which can be found in Data/Raw. Running the flask application will denoise these astronomical
images and allow you to choose a star to track throughout the images. You can see the images after denoising in Data/Testing/Denoised, and
you can see the particles and predictions of the particle filter in Data/Testing/ParticleFilter. The positions for each image are stored as positions.npy

If you want to run this program with your own set of images, delete the images, positions, and .fits files currently in the Data folder and put your images into Data/Raw.

This is still a work in progress, so some errors or unoptimized portions still exist.
