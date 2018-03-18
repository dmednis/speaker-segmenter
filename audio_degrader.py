#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore")  # noqa
import argparse
import re
import librosa as lr
import numpy as np
import subprocess
import tempfile
import logging

logging.basicConfig(level=logging.DEBUG)

TEST_WAV = './test_files/test.wav'

RUBBERBAND_PROCESSINGS = ['time-stretching', 'pitch-shifting']


def mix_with_sound_data(x, z, snr):
    """ Mix x with sound from sound_path

    Args:
        x (numpy array): Input signal
        z (numpy array): Input mix signal
        snr (float): Signal-to-noise ratio
    """
    while z.shape[0] < x.shape[0]: # loop in case noise is shorter than
        z = np.concatenate((z, z), axis=0)
    z = z[0: x.shape[0]]
    rms_z = np.sqrt(np.mean(np.power(z, 2)))
    logging.debug("rms_z: %f" % rms_z)
    rms_x = np.sqrt(np.mean(np.power(x, 2)))
    logging.debug("rms_x: %f" % rms_x)
    snr_linear = 10 ** (snr / 20.0)
    logging.debug("snr , snr_linear: %f, %f" % (snr, snr_linear))
    snr_linear_factor = rms_x / rms_z / snr_linear
    logging.debug("y = x  + z * %f" % snr_linear_factor)
    y = x + z * snr_linear_factor
    rms_y = np.sqrt(np.mean(np.power(y, 2)))
    y = y * rms_x / rms_y
    return y


def mix_with_sound(x, sr, sound_path, snr):
    """ Mix x with sound from sound_path

    Args:
        x (numpy array): Input signal
        sound_path (str): Name of sound
        snr (float): Signal-to-noise ratio
    """
    z, sr = lr.core.load(sound_path, sr=sr, mono=True)
    while z.shape[0] < x.shape[0]: # loop in case noise is shorter than 
        z = np.concatenate((z, z), axis=0)
    z = z[0: x.shape[0]]
    rms_z = np.sqrt(np.mean(np.power(z, 2)))
    logging.debug("rms_z: %f" % rms_z)
    rms_x = np.sqrt(np.mean(np.power(x, 2)))
    logging.debug("rms_x: %f" % rms_x)
    snr_linear = 10 ** (snr / 20.0)
    logging.debug("snr , snr_linear: %f, %f" % (snr, snr_linear))
    snr_linear_factor = rms_x / rms_z / snr_linear
    logging.debug("y = x  + z * %f" % snr_linear_factor)
    y = x + z * snr_linear_factor
    rms_y = np.sqrt(np.mean(np.power(y, 2)))
    y = y * rms_x / rms_y
    return y


def test_mix_with_sound():
    x, sr = lr.core.load(TEST_WAV, mono=True)
    y = mix_with_sound(x, sr, './sounds/white-noise.wav', -6)
    lr.output.write_wav(
        TEST_WAV.replace('.wav', '_wnoise-6.wav'),
        y, sr=sr, norm=False)
    y = mix_with_sound(x, sr, './sounds/white-noise.wav', 20)
    lr.output.write_wav(
        TEST_WAV.replace('.wav', '_wnoise20.wav'),
        y, sr=sr, norm=False)


def convolve(x, sr, ir_path, level=1.0):
    """ Apply convolution to x using impulse response given
    """
    logging.info('Convolving with %s and level %f' % (ir_path, level))
    x = np.copy(x)
    ir, sr = lr.core.load(ir_path, sr=sr, mono=True)
    return np.convolve(x, ir, 'full')[0:x.shape[0]] * level + x * (1 - level)


def test_convolve():
    x, sr = lr.core.load(TEST_WAV, mono=True)
    y = convolve(x, sr, './impulse_responses/ir_classroom.wav', 0.6)
    lr.output.write_wav(
        TEST_WAV.replace('.wav', '_classroom.wav'),
        y, sr=sr, norm=False)
    y = convolve(x, sr, './impulse_responses/ir_smartphone_mic.wav')
    lr.output.write_wav(
        TEST_WAV.replace('.wav', '_smartphone_mic.wav'),
        y, sr=sr, norm=False)


def tmp_path(ext=''):
    tf = tempfile.NamedTemporaryFile()
    return tf.name + ext


def test_tmp_path():
    print(tmp_path())
    print(tmp_path())


def ffmpeg(in_wav, out_wav):
    cmd = ("ffmpeg -y -i {0} -ac 1 " +
           "-acodec pcm_s16le -async 1 {1}").format(
        in_wav, out_wav)
    logging.debug(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print("ERROR!")


def test_ffmpeg():
    ffmpeg(TEST_WAV, TEST_WAV.replace('.wav', '_ffmpeg.wav'))


def lame(in_wav, out_mp3, degree):
    kbps_map = {
        1: 8,
        2: 16,
        3: 24,
        4: 32,
        5: 40
    }
    cmd = "lame -b {0} {1} {2}".format(kbps_map[degree], in_wav,
                                       out_mp3)
    logging.debug(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print("ERROR!")


def test_lame():
    lame(TEST_WAV, TEST_WAV.replace('.wav', '_1.mp3'), 1)
    lame(TEST_WAV, TEST_WAV.replace('.wav', '_2.mp3'), 2)
    lame(TEST_WAV, TEST_WAV.replace('.wav', '_3.mp3'), 3)
    lame(TEST_WAV, TEST_WAV.replace('.wav', '_4.mp3'), 4)
    lame(TEST_WAV, TEST_WAV.replace('.wav', '_5.mp3'), 5)


def apply_mp3(x, sr, degree):
    logging.info("MP3 compression. Degree %d" % degree)
    tmp_file_0 = tmp_path('.wav')
    tmp_file_1 = tmp_path('.wav')
    tmp_file_2 = tmp_path('.mp3')
    tmp_file_3 = tmp_path('.wav')
    lr.output.write_wav(tmp_file_0, x, sr=sr, norm=False)
    ffmpeg(tmp_file_0, tmp_file_1)
    lame(tmp_file_1, tmp_file_2, degree)
    ffmpeg(tmp_file_2, tmp_file_3)
    y, sr = lr.core.load(tmp_file_3, sr=sr, mono=True)
    return y


def apply_gain(x, gain):
    """ Apply gain to x
    """
    logging.info("Apply gain %f dB" % gain)
    x = np.copy(x)
    x = x * (10 ** (gain / 20.0))
    x = np.minimum(np.maximum(-1.0, x), 1.0)
    return x


def trim_beginning(x, nsamples):
    return x[nsamples:]


def test_apply_gain():
    print("Testing apply_gain()")
    # Test file reading
    x, sr = lr.core.load(TEST_WAV, mono=True)
    print("Length of file: %d samples" % len(x))
    print("Mean amplitude before: %f" % np.mean(np.abs(x)))

    y = apply_gain(x, -6)
    print("Mean amplitude after -6dB: %f" % np.mean(np.abs(y)))
    gain_test_wav = TEST_WAV.replace('.wav', '_gain-6.wav')
    lr.output.write_wav(gain_test_wav, y, 8000, norm=False)

    y = apply_gain(x, 6)
    print("Mean amplitude after +6dB: %f" % np.mean(np.abs(y)))
    gain_test_wav = TEST_WAV.replace('.wav', '_gain+6.wav')
    lr.output.write_wav(gain_test_wav, y, 8000, norm=False)


def apply_rubberband(x, sr, time_stretching_ratio=1.0, pitch_shifting_ratio=1.0):
    """ Use rubberband tool to apply time stretching and pitch shifting

    Args:
        x (numpy array): Samples of input signal
        time_stretching_ratio (float): Ratio of time stretching
        pitch_shifting_ratio (float): Ratio of pitch shifting
    Returns:
        (numpy array): Processed audio
    """
    logging.info("Applying rubberband. ts_ratio={0}, ps_ratio={1}".format(
        time_stretching_ratio,
        pitch_shifting_ratio))
    tmp_file_1 = tmp_path()
    tmp_file_2 = tmp_path()
    lr.output.write_wav(tmp_file_1, x, sr=sr, norm=False)
    cmd = "rubberband -c 1 -t {0} -f {1} {2} {3}".format(
        time_stretching_ratio,
        pitch_shifting_ratio,
        tmp_file_1,
        tmp_file_2)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print("ERROR!")
    y, sr = lr.core.load(tmp_file_2, sr=sr, mono=True)
    return y


def test_apply_rubberband():
    x, sr = lr.core.load(TEST_WAV, mono=True)
    t_x = apply_rubberband(x, sr, time_stretching_ratio=0.5)
    p_x = apply_rubberband(x, sr, pitch_shifting_ratio=1.2)
    lr.output.write_wav(TEST_WAV.replace('.wav', '_timestr.wav'),
                        t_x, sr=sr, norm=False)
    lr.output.write_wav(TEST_WAV.replace('.wav', '_pitchshift.wav'),
                        p_x, sr=sr, norm=False)

    
def apply_dr_compression(x, sr, degree):
    tmpfile_1 = tmp_path('.wav')
    tmpfile_2 = tmp_path('.wav')
    lr.output.write_wav(tmpfile_1,
                        x, sr=sr, norm=False)
    if degree == 1:
        cmd = "sox {0} {1} compand 0.01,0.20 -40,-10,-30 5"
    elif degree == 2:
        cmd = "sox {0} {1} compand 0.01,0.20 -50,-50,-40,-30,-40,-10,-30 12"
    elif degree == 3:
        cmd = "sox {0} {1} compand 0.01,0.1 -70,-60,-70,-30,-70,0,-70 45"
    cmd = cmd.format(tmpfile_1, tmpfile_2)
    logging.info(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print("ERROR!")
    y, sr = lr.core.load(tmpfile_2, sr=sr, mono=True)
    return y


def normalize(x, percentage=1.0):
    max_peak = np.max(np.abs(x))
    return x / max_peak * percentage


def test_apply_dr_compression():
    x, sr = lr.core.load(TEST_WAV, mono=True)
    for degree in [1, 2, 3]:
        y = apply_dr_compression(x, sr, degree)
        tmpfile = tmp_path()
        lr.output.write_wav(tmpfile,
                            y, sr=sr, norm=False)
        ffmpeg(tmpfile, TEST_WAV.replace('.wav', '_dr_%d.wav' % degree))


def apply_eq(x, sr, value):
    freq, bw, gain = map(int, value.split('//'))
    logging.info("Equalizing. f=%f, bw=%f, gain=%f" % (freq, bw, gain))
    tmpfile_1 = tmp_path('.wav')
    tmpfile_2 = tmp_path('.wav')
    lr.output.write_wav(tmpfile_1,
                        x, sr=sr, norm=False)
    cmd = "sox {0} {1} equalizer {2} {3} {4}".format(
        tmpfile_1,
        tmpfile_2,
        freq,
        bw,
        gain)
    logging.info(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print("ERROR!")
    y, sr = lr.core.load(tmpfile_2, sr=sr, mono=True)
    return y


def test_apply_eq():
    x, sr = lr.core.load(TEST_WAV, mono=True)
    y = apply_eq(x, sr, '500//50//30')
    tmpfile = tmp_path()
    lr.output.write_wav(tmpfile,
                        y, sr=sr, norm=False)
    ffmpeg(tmpfile, TEST_WAV.replace('.wav', '_eq.wav'))


def test_all():
    test_apply_eq()
    test_apply_dr_compression()
    test_mix_with_sound()
    test_apply_rubberband()
    test_convolve()
    test_tmp_path()
    test_lame()
    test_ffmpeg()
    test_apply_gain()


def main(input_wav, degradations_list, output_wav, testing=False):
    """ Apply degradations to input wav

    Args:
        input_wav (str): Path of input wav
        degradations_list (list): List of degradations (e.g. ['mp3,1'])
        output_wav (str): Path of outpu wav
        testing (bool): True for testing mode
    """
    if testing:
        test_all()
        return
    x, sr = lr.core.load(input_wav, mono=True)
    for degradation in degradations_list:
        degradation_name, value = degradation.split(',')
        if degradation_name == 'mp3':
            x = apply_mp3(x, sr, float(value))
        elif degradation_name == 'gain':
            x = apply_gain(x, float(value))
        elif degradation_name == 'normalize':
            x = normalize(x, float(value))
        elif degradation_name == 'mix':
            sound_path, snr = value.split('//')
            x = mix_with_sound(x, sr, sound_path, float(snr))
        elif degradation_name == 'impulse-response':
            ir_path, level = value.split('//')
            x = convolve(x, sr, ir_path, float(level))
        elif degradation_name == 'time-stretching':
            x = apply_rubberband(x, sr, time_stretching_ratio=float(value))
        elif degradation_name == 'pitch-shifting':
            x = apply_rubberband(x, sr, pitch_shifting_ratio=float(value))
        elif degradation_name == 'dr-compression':
            x = apply_dr_compression(x, sr, degree=float(value))
        elif degradation_name == 'eq':
            x = apply_eq(x, sr, value)
        elif degradation_name == 'start':
            x = x[min(len(x), np.round(sr * float(value))):]
        else:
            logging.warning("Unknown degradation %s" % degradation)
    tmp_file = tmp_path()
    lr.output.write_wav(tmp_file, x, sr=sr, norm=False)
    ffmpeg(tmp_file, output_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Process audio with a sequence of degradations
    Accepted degradadations:
        start,time: Remove audio until start. Value in seconds.
        mp3,quality: Mp3 compression. Value is quality (1-5)
        gain,db: Gain. Value is dB (e.g. gain,-20.3).
        normalize,percentage: Normalize. Percentage in 0.0-1.0 (1.0=full range)
        mix,"sound_path"//snr: Mix with sound at a specified SNR (check sounds folder)
        impulse-response,"impulse_response_path"//level: Apply impulse response (e.g. smartphone sound). Level 0.0-1.0
        dr-compression,degree: Dynamic range compression. Degree can be 1, 2 or 3.
        time-stretching,ratio: Apply time streting. Ratio in from -9.99 to 9.99
        pitch-shifting,ratio: Apply time streting. Ratio in -9.99 to 9.99
        eq,freq_hz//bw_hz//gain_db: Apply equalization with sox.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Note: all audios are transcoded to mono, pcm_s16le (original sample-rate)")

    parser.add_argument('input_wav', metavar='input_wav',
                        type=str,
                        help='Input audio wav')
    parser.add_argument('degradation', metavar='degradation,value',
                        type=str,
                        nargs='*',
                        help='List of sequential degradations')
    parser.add_argument('output_wav', metavar='output_wav',
                        type=str,
                        help='Output audio wav')
    parser.add_argument('--testing', action='store_true',
                        dest='testing',
                        help='Output audio wav')

    args = vars(parser.parse_args())
    main(args['input_wav'],
         args['degradation'],
         args['output_wav'],
         args['testing'])
