'''Note: Some code in tracking portion taken from assignment05 of CSC 630: Computer Vision (fall term 2019)
Source on github: https://github.com/jhargun/Assignment05/blob/master/Assignment05.py'''

import numpy as np
import cv2
import os
import astropy
from astropy.io import fits
from math import ceil, exp, sin, cos, radians
import pickle


'''This just shows the image using opencv's imshow function. It can be useful for testing so I
left it here, but since the images are so large (2048*2048), it is often better to save images
and open them in a different image viewer to analyze them well.'''
def show_image(image, name='Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''This function converts an image to a 0-1 range'''
def normalize_image(data):
    return (data - np.amin(data)) / np.amax(data)


'''This function denoises all the images in the folder provided'''
def noise_reduction(foldername, outFoldername, threshold=.35):
    skipped = 0  # Used to see how many files skipped, error catching

    for i in range(len(os.listdir(foldername)) - 3):
        if i%20 == 0:  # Rudimentary progress bar
            print(i, 'of', len(os.listdir(foldername)))

        path1, path2, path3 = os.listdir(foldername)[i:i+3]
        combinedPath = path1 + path2 + path3  # Used to shorten the next if statement
        # This skips things that aren't images, such as folders, darks (marked by _D_), or autoflats
        if (".fts" not in path1) or (".fts" not in path2) or (".fts" not in path3) or ("_D_" in combinedPath) or ("AutoFlat" in combinedPath):
            skipped += 1
            continue

        # Opens the image and the 2 subsequent images
        with fits.open("{}/{}".format(foldername, path1)) as hdul1:
            with fits.open("{}/{}".format(foldername, path2)) as hdul2:
                with fits.open("{}/{}".format(foldername, path3)) as hdul3:
                    datas = (hdul1[0].data, hdul2[0].data, hdul3[0].data)  # Image data of the 3 files

                    # Median blurring on each image before stacking leads to the best results
                    image = np.zeros(datas[0].shape)
                    for data in datas:
                        im = normalize_image(data)
                        im = np.uint8(im * 255)  # Conversion to uint8 required before medianBlur
                        im = cv2.medianBlur(im, 3)  # Does medianBlur with (3*3) kernel
                        image = np.add(image, im)  # Stacks images

                    # To detect less stars (if there's too much noise), increase threshold; to detect more (not detecting all), decrease
                    image = np.where(image > threshold * np.amax(image), 255, image)  # Chooses pixels with high enough brightness as stars

                    filename = 'Data/{}/Denoised/{}.jpg'.format(outFoldername, path1[:-4])  # [:-4] removes ".fits" from filename
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Makes directory if it doesn't exist
                    cv2.imwrite(filename, (image).astype(int))  # Saves image
    print('Files skipped:', skipped)


'''I used a particle filter to track the stars throughout multiple images. I take a template of the
area surrounding a star, then track that template throughout the series of images.'''
class ParticleFilter(object):
    def __init__(self, frame, template_rect, num_particles, sigma_exp, sigma_dyn):
        self.sigma_exp = sigma_exp
        self.sigma_dyn = sigma_dyn  # High sigma_dyn = more dispersed particles
        t_X = template_rect['x']
        t_Y = template_rect['y']
        height = template_rect['h']
        width = template_rect['w']
        self.template = frame[t_Y:t_Y+height, t_X:t_X+width]  # Makes template using dimensions and position provided

        x = np.random.uniform(t_X, t_X+width, num_particles)  # Uniform distribution of particles over template
        y = np.random.uniform(t_Y, t_Y+height, num_particles)
        self.particles = np.vstack((x, y)).T.astype(int)  # Reshapes array to the correct shape
        self.weights = np.ones(num_particles) / num_particles  # Makes uniform weights

    # This gets the likelyhood that the thing being tracked is at the particle's position
    def get_likelyhood(self, template, frame_cutout):
        # Use mean squared difference to calculate difference between template and cutout at particle's position, higher difference=less likely
        msdif = np.mean((self.template.astype(np.float32) - frame_cutout.astype(np.float32)) ** 2)
        return exp(-1*msdif / (2 * self.sigma_exp ** 2))

    '''Resamples the particles. Particles with a high weight (high likelyhood that the template is located at their position) create many new particles, while
    particles with a low weight (low likelyhood) are less likely to create particles. Therefore, most particles are created near the position of the template.'''
    def resample_particles(self):
        new_parts_x = np.random.choice(self.particles[:, 0], self.particles.shape[0], p=self.weights)  # Makes new particles using old weights
        new_parts_y = np.random.choice(self.particles[:, 1], self.particles.shape[0], p=self.weights)
        new_parts = np.vstack((new_parts_x, new_parts_y)).T
        self.weights = np.ones(len(self.particles)) / len(self.particles)  # Change weights to uniform distribution
        return new_parts

    # This method processes an image and updates the particle filter
    def process(self, frame):
        self.particles = self.resample_particles() + np.random.normal(0, self.sigma_dyn, self.particles.shape)  # Randomizes positions
        self.particles = self.particles.astype(int)

        width = self.template.shape[1]  # Width and height of template
        height = self.template.shape[0]
        f_w = frame.shape[1]  # Width and height of frame
        f_h = frame.shape[0]

        w_2 = ceil(width/2)  # Minimum distance of particles from left and right sides
        h_2 = ceil(height/2)  # Distance from top and bottom
        rand_parts_h = np.random.randint(h_2, f_h-h_2, self.particles.shape[0])  # Creates particles at random positions
        rand_parts_w = np.random.randint(w_2, f_w-w_2, self.particles.shape[0])
        rand_parts = np.vstack((rand_parts_w, rand_parts_h)).T

        # These next 4 lines replace any out of bounds particles with random particles
        self.particles = np.where(np.array([self.particles[:, 0] < w_2]*2).T, rand_parts, self.particles)
        self.particles = np.where(np.array([self.particles[:, 1] < h_2]*2).T, rand_parts, self.particles)
        self.particles = np.where(np.array([self.particles[:, 0] > f_w-w_2]*2).T, rand_parts, self.particles)
        self.particles = np.where(np.array([self.particles[:, 1] > f_h-h_2]*2).T, rand_parts, self.particles)

        for i in range(len(self.particles)):  # Change weights
            x = self.particles[i][0] - w_2
            y = self.particles[i][1] - h_2
            likelyhood = self.get_likelyhood(self.template, frame[y:y+height, x:x+width])
            self.weights[i] = self.weights[i] * likelyhood

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

    # This creates a visual representation of the particle filter
    def render(self, frame_in):
        for p in self.particles:  # Shows particles
            p = tuple(p.astype(int))
            cv2.circle(frame_in, p, 2, (255,0,0), -1)

        # Getting weighted mean and stdev of x and y values
        x_av = np.average(self.particles[:, 0], weights=self.weights)
        y_av = np.average(self.particles[:, 1], weights=self.weights)

        width = self.template.shape[1]  # Width and height of template
        height = self.template.shape[0]

        x = int(x_av - width//2)  # X and y values of the top left of the rectangle
        y = int(y_av - height//2)

        cv2.rectangle(frame_in, (x, y), (x+width, y+height), 255, 2)  # Makes box for predicted location of template

    # Gets the location of the template
    def get_template_location(self):
        x_av = np.average(self.particles[:, 0], weights=self.weights)
        y_av = np.average(self.particles[:, 1], weights=self.weights)
        return (int(x_av), int(y_av))


'''After some tuning, I found that these num_particle, sigma_exp, and sigma_dyn values worked with certain datasets.
This could still be tuned further though.
foldername is directory containing denoised images, outFolder is directory where the output should be saved to
Note: The tracking method used will most likely be changed from particle filter.'''
def track(outFoldername, foldername, template_rect, num_particles=500, sigma_exp=3, sigma_dyn=10):
    skipped = 0
    # print('Length of folder:', len(os.listdir(foldername)))
    path = os.listdir(foldername)[0]  # Makes initial frame for particle filter
    for i in range(len(os.listdir(foldername))):  # Checks to make sure initial file is .jpg
        path = os.listdir(foldername)[i]
        if '.jpg' in path:
            break

    frame = cv2.imread("{}/{}".format(foldername, path))
    # template_rect = {'x': 700, 'y':1000, 'h': 300, 'w': 300}  # Contains coords and height/width of template
    pf = ParticleFilter(frame, template_rect, num_particles, sigma_exp, sigma_dyn)

    positions = []  # This list will hold the x,y positions of the template in each image

    for i, path in enumerate(os.listdir(foldername)):
        if i%20 == 0:  # Rudimentary progress bar
            print(i, 'of', len(os.listdir(foldername)))

        if ".jpg" not in path:  # Skip files that aren't images
            skipped += 1
            continue

        frame = cv2.imread("{}/{}".format(foldername, path))
        pf.process(frame)  # Updates particle filter for next image
        processed = frame.copy()
        pf.render(processed)

        filename = 'Data/{}/ParticleFilter/{}.jpg'.format(outFoldername, path[:-4])  # [:-4] removes ".fits" from filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Makes directory if it doesn't exist
        cv2.imwrite(filename, (processed).astype(int))  # Saves image

        positions.append(pf.get_template_location())

    # No reason to pickle it and save as a .txt file since it isn't human readable
    # with open('Data/{}/positions.txt'.format(outFoldername), "wb") as file:  # Saves positions in txt file
    #     pickle.dump(positions, file)
    np.save('Data/{}/positions.npy'.format(outFoldername), np.array(positions), allow_pickle=True)
    print('Skipped:', skipped)
