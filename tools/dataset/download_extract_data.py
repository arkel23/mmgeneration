from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(
    file_id='184NpGg0wIYWj6KnOj3mN9psZ2r2JX-MZ',
    dest_path='./dafre_faces.tar.gz', unzip=False)

gdd.download_file_from_google_drive(
    file_id='11mcQoIYsjk0N1AA-QftNJ6ngVKt69xia',
    dest_path='./dafre_full.tar.gz', unzip=False)

gdd.download_file_from_google_drive(
    file_id='1bEF1CrWLYfRJYauBY9bpiZ-LMnyEn24w',
    dest_path='./moeimouto_animefacecharacterdataset.tar.gz', unzip=False)

gdd.download_file_from_google_drive(
    file_id='1USvdrXUExzuB1O5z0nDpR74pDNzI0KxL',
    dest_path='./personai_icartoonface_rectrain.zip', unzip=True)

gdd.download_file_from_google_drive(
    file_id='1lUq5-BgNgqj-gIP33XLiLyJL10gQupza',
    dest_path='./personai_icartoonface_rectest.zip', unzip=True)
