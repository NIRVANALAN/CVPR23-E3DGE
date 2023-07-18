import os
import html
import glob
import uuid
import hashlib
import requests
from tqdm import tqdm


ffhq_full_model_spec = dict(file_url='https://drive.google.com/uc?id=13s_dH768zJ3IHUjySVbD1DqcMolqKlBi',
                            alt_url='', file_size=202570217, file_md5='1ab9522157e537351fcf4faed9c92abb',
                            file_path='full_models/ffhq1024x1024.pt',)
ffhq_volume_renderer_spec = dict(file_url='https://drive.google.com/uc?id=1zzB-ACuas7lSAln8pDnIqlWOEg869CCK',
                                 alt_url='', file_size=63736197, file_md5='fe62f26032ccc8f04e1101b07bcd7462',
                                 file_path='pretrained_renderer/ffhq_vol_renderer.pt',)

# all md5 & filesize fixed, waiting for the 
id_loss_model = dict(file_url='https://drive.google.com/uc?id=1yrqdkBoZo3m0JEL1rmrSRvCrmn4-NHxj',
                            alt_url='', file_size=175367323, file_md5='ccee24aae1d36888fc42390ae39d1b36',
                            file_path='pretrained_models/model_ir_se50.pth',)

E3DGE_toonify_generator = dict(file_url='https://drive.google.com/uc?id=1CIibwVOXRIC-UiNsO2IgyTTQ0SmKvh3m',
                            alt_url='', file_size=202570153, file_md5='72af2c8d75d93708b8a0086eacf72ee9',
                            file_path='full_models/Toonify400_1024x1024.pt',)

E3DGE_toonify_encoder = dict(file_url='https://drive.google.com/uc?id=1b_LtIihcgSphx2sIcIknZvZN_YDjYapU',
                            alt_url='', file_size=1168970167, file_md5='df2a9c660499b186ee2c16a20512d8ee',
                            file_path='pretrained_models/Toonify/no_local_basic_trainer.pt',)

E3DGE_FFHQ_model = dict(file_url='https://drive.google.com/uc?id=1ZRGAk-ACh5y-8ZsXLMW885wAv_A8u1Tk',
                            alt_url='', file_size=1582414403, file_md5='4bc643cfa89adba5fa7c0451c49d1c64',
                            file_path='pretrained_models/E3DGE_Full_Runner.pt',)

# E3DGE_FFHQ_model_2dalign = dict(file_url='https://drive.google.com/uc?id=1wFyAl9f2kjFMH6Q6ooZtTbIRt5wfJNI_',
#                             alt_url='', file_size=1575299434, file_md5='c047a6bef3d308d60b2eec50b9272acc',
#                             file_path='pretrained_models/E3DGE_2DAlignOnly_Runner.pt',)



def download_pretrained_models():

    
    # download StyleSDF models first
    print('Downloading FFHQ pretrained volume renderer')
    with requests.Session() as session:
        try:
            download_file(session, ffhq_volume_renderer_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, ffhq_volume_renderer_spec, use_alt_url=True)

    print('Downloading FFHQ full model (1024x1024)')
    with requests.Session() as session:
        try:
            download_file(session, ffhq_full_model_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, ffhq_full_model_spec, use_alt_url=True)


    print('Downloading E3DGE model pretrained on FFHQ.')
    with requests.Session() as session:
        try:
            # download_file(session, E3DGE_FFHQ_model_2dalign)
            download_file(session, E3DGE_FFHQ_model)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            # download_file(session, E3DGE_FFHQ_model_2dalign, use_alt_url=True)
            download_file(session, E3DGE_FFHQ_model, use_alt_url=True)

    print('Downloading E3DGE Toonifiy encoder & generator pretrained on FFHQ.')
    with requests.Session() as session:
        try:
            download_file(session, E3DGE_toonify_generator)
            download_file(session, E3DGE_toonify_encoder)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, E3DGE_toonify_encoder, use_alt_url=True)

    print('Downloading Arcface model.')
    with requests.Session() as session:
        try:
            download_file(session, id_loss_model)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, id_loss_model, use_alt_url=True)
    

def download_file(session, file_spec, use_alt_url=False, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    if use_alt_url:
        file_url = file_spec['alt_url']
    else:
        file_url = file_spec['file_url']

    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    progress_bar = tqdm(total=file_spec['file_size'], unit='B', unit_scale=True)
    print(f'downloading {file_path}...')
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        progress_bar.reset()
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except Exception as e:
            # print(e)
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'confirm=t' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    progress_bar.close()

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

if __name__ == "__main__":
    download_pretrained_models()
