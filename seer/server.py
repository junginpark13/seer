from flask import Flask, render_template, flash, request
from wtforms import DateTimeField, Form, IntegerField, TextField, TextAreaField, validators, SelectField, StringField, SubmitField
from sklearn.externals import joblib
import numpy as np

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# SciKit-Learn related
model = joblib.load('model_rf.pickle')
with open('column_info.txt', 'r') as f:
    column_info = f.read().split()


class ReusableForm(Form):
    reg = SelectField(label='Region:', choices=[
        ('1501', 'San Francisco-Oakland'),
        ('1502', 'Connecticut'),
        ('1520', 'Metropolitan Detroit'),
        ('1521', 'Hawaii'),
        ('1522', 'Iowa'),
        ('1523', 'New Mexico'),
        ('1525', 'Seattle (Puget Sound)'),
        ('1526', 'Utah'),
        ('1527', 'Metropolitan Atlanta'),
        ('1529', 'Alaska'),
        ('1531', 'San Jose-Monterey'),
        ('1535', 'Los Angeles'),
        ('1537', 'Rural Georgia'),
        ('1541', 'Greater California (excluding SF, LA & SJ)'),
        ('1542', 'Kentucky'),
        ('1543', 'Louisiana'),
        ('1544', 'New Jersey'),
        ('1547', 'Greater Georgia (excluding AT and RG)')],
        validators=[validators.required()])
    mar_stat = SelectField(label='Marital status:', choices=[
        ('1', 'Single (never married)'),
        ('2', 'Married (including common law)'),
        ('3', 'Separated'),
        ('4', 'Divorced'),
        ('5', 'Widowed'),
        ('6', 'Unmarried or domestic partner (same sex or opposite sex or unregistered)')],
        validators=[validators.required()])
    race1v = SelectField(label='Race:', choices=[
        ('01', 'White'),
        ('02', 'Black'),
        ('03', 'American Indian, Aleutian, Alaskan Native or Eskimo (includes all indigenous populations of the Western hemisphere)'),
        ('04', 'Chinese'),
        ('05', 'Japanese'),
        ('06', 'Filipino'),
        ('07', 'Hawaiian'),
        ('08', 'Korean'),
        ('10', 'Vietnamese'),
        ('11', 'Laotian'),
        ('12', 'Hmong'),
        ('13', 'Kampuchean (including Khmer and Cambodian)'),
        ('14', 'Thai'),
        ('15', 'Asian Indian or Pakistani, NOS'),
        ('16', 'Asian Indian'),
        ('17', 'Pakistani'),
        ('20', 'Micronesian, NOS'),
        ('21', 'Chamorran'),
        ('22', 'Guamanian, NOS'),
        ('25', 'Polynesian, NOS'),
        ('26', 'Tahitian'),
        ('27', 'Samoan'),
        ('28', 'Tongan'),
        ('30', 'Melanesian, NOS'),
        ('31', 'Fiji Islander'),
        ('32', 'New Guinean'),
        ('96', 'Other Asian, including Asian, NOS and Oriental, NOS'),
        ('97', 'Pacific Islander, NOS'),
        ('98', 'Other')],
        validators=[validators.required()])
    nhiade = SelectField(label='Hispanic origin:', choices=[
        ('0', 'Non-Spanish-Hispanic-Latino'),
        ('1', 'Mexican'),
        ('2', 'Puerto Rican'),
        ('3', 'Cuban'),
        ('4', 'South or Central American excluding Brazil'),
        ('5', 'Other specified Spanish/Hispanic Origin including Europe'),
        ('6', 'Spanish/Hispanic/Latino, NOS'),
        ('7', 'NHIA Surname Match Only'),
        ('8', 'Dominican Republic')],
        validators=[validators.required()])
    sex = SelectField(label='Hispanic origin:', choices=[
        ('1', 'Male'),
        ('2', 'Female')],
        validators=[validators.required()])
    age_dx = IntegerField(label='Age:', validators=[validators.required()])
    mdxrecmp = IntegerField(label='Diagnosis Month:', validators=[validators.required()])
    year_dx = IntegerField(label='Diagnosis Year:', validators=[validators.required()])
    beho3v = SelectField(label='Malignancies:', choices=[
        # ('0', 'Benign (Reportable for intracranial and CNS sites only)'),
        ('1', 'Uncertain whether benign or malignant, borderline malignancy, low malignant potential, and uncertain malignant potential'),
        ('2', 'Carcinoma in situ; noninvasive'),
        ('3', 'Malignant, primary site; invasive')],
        validators=[validators.required()])
    lateral = SelectField(label='Side of tumor orgin:', choices=[
        ('0', 'Not a paired site'),
        ('1', 'Right: origin of primary'),
        ('2', 'Left: origin of primary'),
        ('3', 'Only one side involved, right or left origin unspecified'),
        ('4', 'Bilateral involvement, lateral origin unknown; stated to be single primary'),
        ('5', 'Paired site: midline tumor')],
        validators=[validators.required()])
    grade = SelectField(label='Tumor grade:', choices=[
        ('1', 'Grade I; grade i; grade 1; well differentiated; differentiated, NOS'),
        ('2', 'Grade II; grade ii; grade 2; moderately differentiated; moderately differentiated; intermediate differentiation'),
        ('3', 'Grade III; grade iii; grade 3; poorly differentiated; differentiated'),
        ('4', 'Grade IV; grade iv; grade 4; undifferentiated; anaplastic'),
        ('5', 'T-cell; T-precursor'),
        ('6', 'B-cell; Pre-B; B-Precursor'),
        ('7', 'Null cell; Non T-non B'),
        ('8', 'N K cell (natural killer cell)')],
        validators=[validators.required()])

@app.route('/', methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)

    # print form.errors
    if request.method == 'POST':
        # Run prediction
        num_features = len(column_info)
        X = np.zeros((1, num_features), dtype=np.float32)

        reg = int(request.form['reg'])
        reg_idx = column_info.index('REG__' + str(reg))
        X[0, reg_idx] = 1.0
        print('reg_idx', reg_idx)

        mar_stat = int(request.form['mar_stat'])
        mar_stat_idx = column_info.index('MAR_STAT__' + str(mar_stat))
        X[0, mar_stat_idx] = 1.0
        print('mar_stat_idx', mar_stat_idx)

        race1v = int(request.form['race1v'])
        race1v_idx = column_info.index('RACE1V__' + str(race1v))
        X[0, race1v_idx] = 1.0
        print('race1v_idx', race1v_idx)

        nhiade = int(request.form['nhiade'])
        nhiade_idx = column_info.index('NHIADE__' + str(nhiade))
        X[0, nhiade_idx] = 1.0
        print('nhiade_idx', nhiade_idx)

        sex = int(request.form['sex'])
        sex_idx = column_info.index('SEX__' + str(sex))
        X[0, sex_idx] = 1.0
        print('sex_idx', sex_idx)

        age_dx = int(request.form['age_dx'])
        age_dx_idx = column_info.index('AGE_DX')
        X[0, age_dx_idx] = age_dx
        print('age_dx_idx', age_dx_idx)

        mdxrecmp = int(request.form['mdxrecmp'])
        mdxrecmp_idx = column_info.index('MDXRECMP')
        X[0, mdxrecmp_idx] = mdxrecmp
        print('mdxrecmp_idx', mdxrecmp_idx)

        year_dx = int(request.form['year_dx'])
        year_dx_idx = column_info.index('YEAR_DX')
        X[0, year_dx_idx] = year_dx
        print('year_dx_idx', year_dx_idx)

        beho3v = int(request.form['beho3v'])
        beho3v_idx = column_info.index('BEHO3V__' + str(beho3v))
        X[0, beho3v_idx] = 1.0
        print('beho3v_idx', beho3v_idx)

        lateral = int(request.form['lateral'])
        lateral_idx = column_info.index('LATERAL__' + str(lateral))
        X[0, lateral_idx] = 1.0
        print('lateral_idx', lateral_idx)

        grade = int(request.form['grade'])
        grade_idx = column_info.index('GRADE__' + str(grade))
        X[0, grade_idx] = 1.0
        print('grade_idx', grade_idx)

        print(X)

        y = model.predict(X)
        y_prob = model.predict_proba(X)

        if form.validate():
            flash('5-year survival probability after surgery: {} %'.format(y_prob[0][1] * 100))
        else:
            flash('Error: All the form fields are required. ')

    return render_template('hello.html', form=form)


if __name__ == '__main__':
    app.run()
