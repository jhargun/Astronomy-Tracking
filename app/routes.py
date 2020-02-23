from app import application
from flask import render_template, request, redirect, make_response
from app.forms import CanvasForm, HomepageForm, NoiseForm
import shutil

from process import *

'''These are major TODOs:
TODO: fix whatever's wrong with the function that changes denoising threshold (possibly delete folders?)
'''
'''Note: Be warned that the wait times for the pages to load will likely be long.'''


'''Helper method, denoises an image set and creates the initial image that will be rendered on the canvas.'''
def denoise(threshold):
    noise_reduction('Data/Raw', 'Testing', threshold=threshold)  # Denoises images
    path = os.listdir('Data/Testing/Denoised/')[0]
    init_pic = cv2.imread('Data/Testing/Denoised/' + path)
    # Puts the initial image into the app folder, where the html can see it to render the image
    cv2.imwrite('app/static/images/Initial_Image.jpg', init_pic)  # Initial_Image.jpg is the filename that canvas.html looks for



'''TODO: Allow user to upload image set to app rather than needing to move it to specific directory'''

'''Here, I will make a homepage with a button that begins the denoising process and another that redirects
to a help page with instructions'''
@application.route('/', methods=['GET', 'POST'])
def home():
    form = HomepageForm()
    if form.validate_on_submit():
        # print('validated')
        # If user says to denoise, folder for denoised images doesn't exist, or folder is empty: denoise
        if form.data['denoise'] or (not os.path.isdir('Data/Raw') or len(os.listdir('Data/Raw')) == 0):
            denoise(.35)
        # print('done')
        return redirect('/track')  # Redirects to tracking step
    return render_template('home.html', form=form)


'''This is the page where the user selects a star to track'''
@application.route('/track', methods=['GET', 'POST'])
def canvas():
    form = CanvasForm()
    form2 = NoiseForm()

    # If post request with form1, tracks the star selected
    if form.validate_on_submit() and form.submit.data:
        x = form.data['x']
        y = form.data['y']
        # outFolder = form.data['outputFolder']
        '''TODO: Account for cases where x and y are close to edges
        Note: Done, but can probably be made more pythonic
        TODO: UNHARDCODE STUFF!!!'''
        # This part makes the template bounds so that the star is in the center
        template_rect = {'x': x - 100, 'y': y - 100, 'h': 200, 'w': 200}
        if x < 100 or x > 1948:  # I hard coded a width bound here, change in the future
            template_rect['x'] = x
            template_rect['w'] = 2 * x
            if x > 1948:
                template_rect['x'] = 2048 - x
        if y < 100 or y > 1948:  # I hard coded a height bound here, change in the future
            template_rect['y'] = y
            template_rect['h'] = 2 * y
            if y > 1948:
                template_rect['y'] = 2048 - y

        '''TODO: I'm leaving previous code, which makes a dictionary, as is for now. Change this in the future'''
        x = template_rect['x']
        y = template_rect['y']
        height = template_rect['h']
        width = template_rect['w']
        img = cv2.imread('app/static/images/Initial_Image.jpg', cv2.IMREAD_GRAYSCALE)
        template = img[y:y+height, x:x+width]
        track_matrix_fast('Testing', 'Data/Testing/Denoised', template, img.shape)
        # track_matrix('Testing', 'Data/Testing/Denoised', template_rect)
        # track('Testing', 'Data/Testing/Denoised', template_rect)
        return redirect('/')

    # If post request with form2, adjusts the denoising parameter
    elif form2.validate_on_submit() and form2.submit.data:
        response = make_response(render_template('canvas.html', form=form, form2=form2))

        if form2.data['adjust'] == 'reset':
            response.set_cookie('threshold', '.35')  # If user chooses reset, resets threshold to .35
            denoise(.35)  # Denoises with .35 as threshold
        else:
            # Otherwise, increases/decreases threshold by amount specified by user
            threshold = round(float(request.cookies.get('threshold')) + float(form2.data['adjust']), 2)  # Rounds to 2 decimals
            print(str(threshold))  # Just for testing
            denoise(threshold)  # Denoises with new threshold
            response.set_cookie('threshold', str(threshold))  # Sets new threshold

        return response

    response = make_response(render_template('canvas.html', form=form, form2=form2))
    if 'threshold' not in request.cookies:  # Sets cookie holding threshold if it hasn't been set
        response.set_cookie('threshold', '.35')  # Note: Threshold has to be a string, can't use a float
    return response


'''At the moment, all this does is detect lines. I will implement more troubleshooting in the future'''
@application.route('/troubleshoot')
def troubleshoot():
    outdir = 'app/static/images/Line_Detection/'  # Directory where images stored
    shutil.rmtree(outdir)  # Deletes any old images with lines in them
    os.makedirs(os.path.dirname(outdir + 'image.jpg'), exist_ok=True)  # Remakes folder (now empty)
    line_detect(outdir, 'Data/Testing/Denoised/')  # Detects images with lines
    paths = os.listdir(outdir)
    print(paths[0])
    paths = ["static/images/Line_Detection/" + path for path in paths]
    return render_template('troubleshoot.html', paths=paths)
