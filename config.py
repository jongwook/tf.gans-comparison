from models import *


model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'WGAN-GP128', 'EBGAN', 'BEGAN', 'BEGAN128', 'DRAGAN', 'DRAGAN128', 'CoulombGAN']

def get_model(mtype, name, training):
    model = None
    if mtype == 'DCGAN':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        model = wgan.WGAN
    elif mtype == 'WGAN-GP':
        model = wgan_gp.WGAN_GP
    elif mtype == 'WGAN-GP128':
        model = wgan_gp128.WGAN_GP128
    elif mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        model = began.BEGAN
    elif mtype == 'BEGAN128':
        model = began128.BEGAN128
    elif mtype == 'DRAGAN':
        model = dragan.DRAGAN
    elif mtype == 'DRAGAN128':
        model = dragan128.DRAGAN128
    elif mtype == 'COULOMBGAN':
        model = coulombgan.CoulombGAN
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training)


def get_dataset(dataset_name):
    celebA_64 = './data/celebA_tfrecords/*.tfrecord'
    celebA_128 = './data/celebA_128_tfrecords/*.tfrecord'
    spectrograms_128 = './data/spectrograms/*.tfrecord'
    lsun_bedroom_128 = './data/lsun/bedroom_128_tfrecords/*.tfrecord'

    if dataset_name == 'celeba':
        path = celebA_128
        n_examples = 202599
    elif dataset_name == 'lsun':
        path = lsun_bedroom_128
        n_examples = 3033042
    elif dataset_name == 'nsynth':
        path = spectrograms_128
        n_examples = 289205
    else:
        raise ValueError('{} is does not supported. dataset must be celeba or lsun.'.format(dataset_name))

    return path, n_examples


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

