from app import application
from flask import render_template, request
from app.forms import CanvasForm

from process import *


'''Note: At the moment, before rendering the page, I run denoising on all the images. I will
change this in the future as I add functionality such as buttons to choose the denoising parameters,
but for now, be warned that the wait times for the page to load will likely be long. I added a simple
print statement that tracks the progress of the denoising function. Similar wait times will be
encountered when submitting a post request due to the time required to process the particle filter.'''

@application.route('/', methods=['GET', 'POST'])
# @application.route('/track')
def canvas():
    form = CanvasForm()
    # If post request, denoises and tracks the star selected
    if form.validate_on_submit():
        x = form.data['x']
        y = form.data['y']
        # outFolder = form.data['outputFolder']
        '''TODO: Account for cases where x and y are close to edges'''
        template_rect = {'x': x - 100, 'y': y - 100, 'h': 200, 'w': 200}
        track('Testing', 'Data/Testing/Denoised', template_rect)

    # If get request, finds the additional image to be displayed by the flask app
    noise_reduction('Data/Raw', 'Testing')  # Denoises images
    path = os.listdir('Data/Testing/Denoised/')[0]
    init_pic = cv2.imread('Data/Testing/Denoised/' + path)
    # Puts the initial image into the app folder, where the html can see it to render the image
    cv2.imwrite('app/static/images/Initial_Image.jpg', init_pic)  # Initial_Image.jpg is the filename that canvas.html looks for
    return render_template('canvas.html', form=form)
