from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os

def synchronize_dir(drive, drive_dir, local_dir):
    print('Synchronizing %s with %s...' % (drive_dir, local_dir))
    drive_items = [{ 'id': item['id'], 'title': item['title'], 'mimeType': item['mimeType']} for item in drive.ListFile({'q': "'%s' in parents and trashed=false" % drive_dir['id']}).GetList()]
    local_items = os.listdir(local_dir)
    local_items_to_upload = [item for item in os.listdir(local_dir) if item not in map(lambda x: x['title'], drive_items)]
  
    for local_item_to_upload in local_items_to_upload:
        local_item_to_upload_abs = os.path.join(local_dir, local_item_to_upload)
        if os.path.isfile(local_item_to_upload_abs):
            print('Creating file %s...' % local_item_to_upload_abs)
            file = drive.CreateFile({ 'parents': [{ 'kind': 'drive#childList', 'id': drive_dir['id']}], 'title': local_item_to_upload })
            file.SetContentFile(local_item_to_upload_abs)
            file.Upload()
        elif os.path.isdir(local_item_to_upload_abs):
            print('Creating directory %s...' % local_item_to_upload_abs)
            directory = drive.CreateFile({ 'parents': [{ 'kind': 'drive#childList', 'id': drive_dir['id']}], 'title': local_item_to_upload, 'mimeType': 'application/vnd.google-apps.folder' })
            directory.Upload()
      
    for local_subdir in [item for item in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, item))]:
        child_drive_dir = [{ 'id': item['id'], 'title': item['title']} for item in drive.ListFile({'q': "'%s' in parents and title='%s' and trashed=false" % (drive_dir['id'], local_subdir)}).GetList()][0]
        child_local_dir = os.path.join(local_dir, local_subdir)
    
        synchronize_dir(drive, child_drive_dir, child_local_dir)
        
def synchronize_all(drive):
    # Authenticate and create the PyDrive client.
    # This only needs to be done once in a notebook.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    drive_root_dirs = [{ 'id': item['id'], 'title': item['title']} for item in drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList() if item['title'] in ['nn-nbirds']][0]
    drive_sync_dirs = [{ 'id': item['id'], 'title': item['title']} for item in drive.ListFile({'q': "'%s' in parents and trashed=false" % drive_root_dirs['id']}).GetList() if item['title'] in ['logs', 'cnn_models']]

    # TODO : Create directories structure
    # /nn-nbirds
    #   /cnn_models
    #   /logs
  
    for drive_sync_dir in drive_sync_dirs:
        synchronize_dir(drive, drive_sync_dir, drive_sync_dir['title'])