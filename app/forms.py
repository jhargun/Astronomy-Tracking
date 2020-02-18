from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, HiddenField, StringField, BooleanField, RadioField
from wtforms.validators import DataRequired


'''This is just a box where the user marks whether they want to denoise a series of
images and a button that begins the denoising process if they chose to denoise before
redirecting to the tracking page'''
class HomepageForm(FlaskForm):
    denoise = BooleanField(default="checked")  # By default, it is true
    submit = SubmitField('Go to tracking')


class NoiseForm(FlaskForm):
    # I might want to name the 2nd one something else, not sure if people will know what noise is
    adjust = RadioField('adjust', choices=[('-.05', 'Star not detected'), ('.05', 'Too much noise'), ('reset', 'Reset to initial settings')])
    # In the above field, -.01 and .01 tell me whether to decrease or increase the threshold respectively
    # threshold = HiddenField('threshold')  # This hidden field holds the threshold used last
    submit = SubmitField('Rerun denoising')


'''
# This is just for testing, for some reason the hidden field isn't working
class NoiseForm2(FlaskForm):
    # I might want to name the 2nd one something else, not sure if people will know what noise is
    adjust = RadioField('adjust', choices=[('-.01', 'Star not detected'), ('.01', 'Too much noise')])
    # In the above field, -.01 and .01 tell me whether to decrease or increase the threshold respectively
    threshold = HiddenField('threshold')  # This hidden field holds the threshold used last
    submit = SubmitField('Rerun denoising')
'''


class CanvasForm(FlaskForm):
    # DataRequired validator makes it so user must click a star
    x = IntegerField('x', validators=[DataRequired()])
    y = IntegerField('y', validators=[DataRequired()])
    '''I will implement this field shortly. It will allow the user to chose the
    name of the output folder'''
    # outputFolder = StringField('outputFolder', validators=[DataRequired()])
    submit = SubmitField('Track')
