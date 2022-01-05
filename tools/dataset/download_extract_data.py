import argparse
from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces', action='store_true', help='dl dafre faces')
    parser.add_argument('--full', action='store_true', help='dl dafre full')
    parser.add_argument('--moe', action='store_true', help='dl moeimouto')
    parser.add_argument('--icf_train', action='store_true', help='dl cf train')
    parser.add_argument('--icf_test', action='store_true', help='dl cf test')

    parser.add_argument('--ckpt_sketch', action='store_true',
                        help='sketchkeras ckpt')
    parser.add_argument('--ckpt_aoda', action='store_true',
                        help='AODA ckpt')

    args = parser.parse_args()

    if args.faces:
        gdd.download_file_from_google_drive(
            file_id='184NpGg0wIYWj6KnOj3mN9psZ2r2JX-MZ',
            dest_path='./dafre_faces.tar.gz', unzip=False)
    if args.full:
        gdd.download_file_from_google_drive(
            file_id='11mcQoIYsjk0N1AA-QftNJ6ngVKt69xia',
            dest_path='./dafre_full.tar.gz', unzip=False)
    if args.moe:
        gdd.download_file_from_google_drive(
            file_id='1bEF1CrWLYfRJYauBY9bpiZ-LMnyEn24w',
            dest_path='./moeimouto_animefacecharacterdataset.tar.gz',
            unzip=False)
    if args.icf_train:
        gdd.download_file_from_google_drive(
            file_id='1USvdrXUExzuB1O5z0nDpR74pDNzI0KxL',
            dest_path='./personai_icartoonface_rectrain.zip', unzip=True)
    if args.icf_test:
        gdd.download_file_from_google_drive(
            file_id='1lUq5-BgNgqj-gIP33XLiLyJL10gQupza',
            dest_path='./personai_icartoonface_rectest.zip', unzip=True)

    if args.ckpt_sketch:
        gdd.download_file_from_google_drive(
            file_id='1Zo88NmWoAitO7DnyBrRhKXPcHyMAZS97',
            dest_path='./model.pth', unzip=False)
    if args.ckpt_aoda:
        gdd.download_file_from_google_drive(
            file_id='1RILKwUdjjBBngB17JHwhZNBEaW4Mr-Ml',
            dest_path='./model.pth', unzip=False)


if __name__ == '__main__':
    main()
