'''Note: Some code in tracking portion taken from assignment05 of CSC 630: Computer Vision (fall term 2019)
Source on github: https://github.com/jhargun/Assignment05/blob/master/Assignment05.py'''

import numpy as np
import cv2
import os
# import astropy
from astropy.io import fits
from math import ceil, floor, exp, sin, cos, radians
# import pickle
# from PIL import Image
from tqdm import tqdm


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


'''TODO: Add the option for a user to change how much the threshold is being changed by'''

'''This function denoises all the images in the folder provided'''
def noise_reduction(foldername, outFoldername, threshold=.35):
    skipped = 0  # Used to see how many files skipped, error catching

    for i in tqdm(range(len(os.listdir(foldername)) - 3)):
        # for i in range(len(os.listdir(foldername)) - 3):
        '''if i%20 == 0:  # Rudimentary progress bar
            print(i, 'of', len(os.listdir(foldername)))'''

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

    # for i, path in enumerate(os.listdir(foldername)):
    for path in tqdm(os.listdir(foldername)):
        '''if i%20 == 0:  # Rudimentary progress bar
            print(i, 'of', len(os.listdir(foldername)))'''

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


'''Detects lines in the denoised images'''
def line_detect(outFoldername, foldername):
    skipped = 0

    # Goes through each image, if line detected saves image in a folder for user examination
    for i, path in enumerate(os.listdir(foldername)):
        if i%20 == 0:  # Rudimentary progress bar
            print(i, 'of', len(os.listdir(foldername)))

        if ".jpg" not in path:  # Skip files that aren't images
            skipped += 1
            continue

        image = cv2.imread("{}/{}".format(foldername, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (3,3), 0)  # Double blurring gave me the best results
        # canny = cv2.Canny(blur, 30, 60)  # Canny image shows edges
        # canny = cv2.Canny(image, 20, 40)

        # lines = cv2.HoughLines(canny, 1, np.pi/180, 200)  # Finds lines in the image
        lines = cv2.HoughLines(blur, 1, np.pi/180, 200)  # Finds lines in the image
        if lines is not None:
            # If lines detected, puts image where the app can access it
            cv2.imwrite("{}/{}".format(outFoldername, path), image)


'''Note: At the moment, I have not fully used matrices. I instead mix for loops with opencv and PIL functions.
This takes some time, so in the future I may want to change this'''
def track_matrix(outFoldername, foldername, template_rect, rot_step=5):
    skipped = 0
    path = os.listdir(foldername)[0]  # Makes initial frame for particle filter

    for i in range(len(os.listdir(foldername))):  # Checks to make sure initial file is .jpg
        path = os.listdir(foldername)[i]
        if '.jpg' in path:
            break

    frame = cv2.imread("{}/{}".format(foldername, path))

    t_X = template_rect['x']
    t_Y = template_rect['y']
    height = template_rect['h']
    width = template_rect['w']
    template = frame[t_Y:t_Y+height, t_X:t_X+width]  # Makes template using dimensions and position provided

    '''Necessary to avoid parts of the template getting cut off'''
    template = cv2.copyMakeBorder(template, t_Y, frame.shape[0] - (t_Y + height), t_X, frame.shape[1] - (t_X + width), cv2.BORDER_CONSTANT, 0)
    # print(template.shape, frame.shape)

    template = Image.fromarray(template)
    print(type(template.rotate(10)))
    templates = [np.array(template.rotate(rot_step*i)) for i in range(360//rot_step)]

    positions = []  # This list will hold the x,y positions of the template in each image

    for i, path in enumerate(os.listdir(foldername)):
        print(i)
        # if i%20 == 0:  # Rudimentary progress bar
        #     print(i, 'of', len(os.listdir(foldername)))

        if ".jpg" not in path:  # Skip files that aren't images
            skipped += 1
            continue

        frame = cv2.imread("{}/{}".format(foldername, path))

        # for temp in templates[:2]:
        #     show_image(temp)

        ''' Note: This doesn't work when the template is the size of the image'''
        probs = np.array([cv2.matchTemplate(frame, template, cv2.TM_CCOEFF) for template in templates])
        detections = np.array([[np.amax(prob), np.argmax(prob)] for prob in probs])
        orientation = np.argmax(detections[:, 0])
        positions.append([detections[orientation][1], orientation])
        # detections = np.array([(np.amax(cv2.matchTemplate(frame, template, cv2.TM_CCOEFF)), np.argmax() for template in templates])
        # orientation = directions.index(max(directions))

    np.save('Data/{}/positions.npy'.format(outFoldername), np.array(positions), allow_pickle=True)
    print('Skipped:', skipped)


'''Makes the initial transformation matrix, long, assumes (x,y) corresponds to top left)
rot_step is how many degrees rotation per step, translate step is how many pixels per translation'''
def make_transform_matrix(img_shape, template_shape, rot_step=5, translate_step=5):
    transform_matrix = []
    for ang in tqdm(range(360 // rot_step)):  # Multiply ang by rot_step to get actual angle
        theta = radians(ang * rot_step)  # Angle is in radians for use later
        transform_matrix.append([])  # Creates new row of transform_matrix for this rotation

        # Top/bottom left/right points of template; NOTE: USE (x,y), z=1 for use with 3*3 matrix
        t_l = np.array([0, 0, 1])
        t_r = np.array([template_shape[1], 0, 1])
        b_l = np.array([0, template_shape[0], 1])
        b_r = np.array([template_shape[1], template_shape[0], 1])
        old_points = np.array([t_l, t_r, b_l, b_r])  # Need a list of old and new points for the homography

        # Finds x, y adjustment for new shape (since rotating, shape will be different from template_shape)
        rotation_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # Matrix for rotation only
        adjust_list = np.array([np.matmul(rotation_matrix, point[:2]) for point in old_points])  # Gets list of rotated points
        # print('Adjust list shape:', adjust_list.shape)
        min_x = ceil(np.amin(adjust_list[:, 0]))
        width = floor(np.amax(adjust_list[:, 0]) - min_x)
        min_y = ceil(np.amin(adjust_list[:, 1]))
        height = floor(np.amax(adjust_list[:, 1]) - min_y)


        # for x in range(0, img_shape[1] - template_shape[1], translate_step):
        #     for y in range(0, img_shape[0] - template_shape[0], translate_step):

        # Iterates through all x and y values possible
        for x in range(-min_x, img_shape[1] - width, translate_step):
            for y in range(-min_y, img_shape[0] - height, translate_step):
                # This is the matrix that does the rotation and translation
                matrix = np.array([[cos(theta), -sin(theta), x],
                                   [sin(theta), cos(theta),  y],
                                   [0,          0,           1]])

                new_points = np.array([np.matmul(matrix, point)[:2] for point in old_points]).astype(int)  # Get new points

                # print('New points:', new_points)

                homography, _ = cv2.findHomography(old_points[:, :2], new_points)  # Homography matrix that will be used later
                transform_matrix[ang].append(homography)
                # if x == -min_x and y == -min_y:
                #     print('Homography:', homography)
                # transform_matrix[ang].append(matrix)

    max_length = 0
    for i in range(len(transform_matrix)):
        if len(transform_matrix[i]) > max_length:
            max_length = len(transform_matrix[i])
    for i in range(len(transform_matrix)):
        transform_matrix[i].append([[0,0,0],[0,0,0],[0,0,0]])

    # Save the matrix for these steps, image size, and template size; saves a lot of time in the future; uses parameters for name
    filename = 'Matrices/{}x{}_{}x{}_{}_{}.npy'.format(img_shape[1], img_shape[0], template_shape[1], template_shape[0], rot_step, translate_step)
    if not os.path.isfile(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Makes matrices folder if it doesn't exist
    np.save(filename, np.array(transform_matrix))  # Saves matrix for future use


'''TODO: Add some troubleshooting where it warns if the similarity between matrices is very small'''


'''Tracks using matrices, much faster than old version
Note: foldername should be where the denoised images are stored
Set save_images to false if you don't want the visualization'''
def track_matrix_fast(outFoldername, foldername, template, img_shape, rot_step=5, translate_step=3, save_images=True):
    template_shape = template.shape

    filename = 'Matrices/{}x{}_{}x{}_{}_{}.npy'.format(img_shape[1], img_shape[0], template_shape[1], template_shape[0], rot_step, translate_step)
    if not os.path.isfile(filename):  # Checking for matrix means if same parameters were already used, no need to remake it
        print('Making new homography matrices')
        make_transform_matrix(img_shape, template_shape, rot_step, translate_step)

    homog_matrices = np.load(filename)
    print('Beginning 2nd part')
    homog_shape = homog_matrices.shape
    print(homog_matrices.shape)
    homog_matrices = np.reshape(np.prod(homog_matrices[:-2]), *homog_matrices[-2:])  # Reshapes it so it works with for loop
    print(homog_matrices.shape)

    background = np.zeros(img_shape)  # Creates black background for template
    positions = []  # Stores positions of template with highest
    skipped = 0

    size = (background.shape[1], background.shape[0])
    templates = []
    for homography in tqdm(homog_matrices):
        templates.append(cv2.warpPerspective(template, homography, size, background, borderMode=cv2.BORDER_TRANSPARENT))
    templates = np.array(templates)
    print(templates.shape)  # Just for testings
    # templates.reshape(homog_matrices.shape)  # Reshapes back to original shape, but now it's the rotated template
    # print(templates.shape)


    for path in tqdm(os.listdir(foldername)):
        if ".jpg" not in path:  # Skip files that aren't images
            skipped += 1
            continue
        image = cv2.imread("{}/{}".format(foldername, path))


        products = np.dot(templates, image)
        print(products.shape)

    np.save('Data/{}/positions.npy'.format(outFoldername), np.array(positions))
    print(skipped)
