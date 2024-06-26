import scipy


def save_wav(out_dir, out_name, rate, data):
    scipy.io.wavfile.write(out_dir + "/" + out_name, rate=rate, data=data)


def save_batch(out_dir, out_name, rate, batch):
    for i, sample in enumerate(batch):
        save_wav(out_dir, out_name[i], rate, sample)
