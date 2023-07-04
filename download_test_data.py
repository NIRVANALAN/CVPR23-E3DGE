import os
import requests

from download_models import download_file

celebahq_testset = dict(file_url='https://drive.google.com/uc?id=1NjGQWrugExRD_lAp06lODrlfNT5zsD3Y',
                            alt_url='', file_size=255479196, file_md5='a44cc5b2d6827169e2a874dd170374ac',
                            file_path='datasets/CelebAMask-HQ-test_img.tar.gz',)

def download_celebahq_testset():
    print('Downloading CelebA-HQ Test set...')
    with requests.Session() as session:
        try:
            download_file(session, celebahq_testset)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, celebahq_testset, use_alt_url=True)
    os.system('tar xzvf datasets/CelebAMask-HQ-test_img.tar.gz -C datasets')
    os.system('rm datasets/CelebAMask-HQ-test_img.tar.gz')


if __name__ == "__main__":
    download_celebahq_testset()
