import os
import requests

from download_models import download_file


E3DGE_FFHQ_model_2dalign = dict(file_url='https://drive.google.com/uc?id=1wFyAl9f2kjFMH6Q6ooZtTbIRt5wfJNI_',
                            alt_url='', file_size=1575299434, file_md5='c047a6bef3d308d60b2eec50b9272acc',
                            file_path='pretrained_models/E3DGE_2DAlignOnly_Runner.pt',)



def download_pretrained_models():


    print('Downloading E3DGE model pretrained on FFHQ.')
    with requests.Session() as session:
        try:
            download_file(session, E3DGE_FFHQ_model_2dalign)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, E3DGE_FFHQ_model_2dalign, use_alt_url=True)

if __name__ == "__main__":
    download_pretrained_models()
