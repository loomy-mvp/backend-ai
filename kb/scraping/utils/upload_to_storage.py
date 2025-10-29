# import the necessary libraries
from google.cloud.exceptions import NotFound, Conflict, Forbidden, BadRequest

CONTENT_TYPE_MAP = {
    # Documents
    "pdf": "application/pdf",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain",
    "rtf": "application/rtf",
    "csv": "text/csv",

    # Data / Markup
    "xml": "application/xml",
    "xsd": "application/xml",
    "json": "application/json",
    "yaml": "application/x-yaml",
    "yml": "application/x-yaml",

    # Archives / Compression
    "zip": "application/zip",
    "gz": "application/gzip",
    "tar": "application/x-tar",
    "rar": "application/vnd.rar",
    "7z": "application/x-7z-compressed",

    # Web
    "html": "text/html",
    "htm": "text/html",
    "css": "text/css",
    "js": "application/javascript",
    "jsonld": "application/ld+json",

    # Images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "webp": "image/webp",
    "ico": "image/x-icon",

    # Audio / Video
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "webm": "video/webm",
}

def _get_bucket(storage_client, bucket_name):
    """
    Get GCS bucket, create if it doesn't exist
    """
    # Try to get the bucket — this avoids an explicit .exists() check
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        print(f"Bucket '{bucket_name}' not found. Creating it...")
        try:
            bucket = storage_client.create_bucket(bucket_name)
            print(f"✅ Created bucket '{bucket_name}'")
        except Conflict:
            # Another process might have created it simultaneously
            bucket = storage_client.get_bucket(bucket_name)
        except Forbidden:
            raise PermissionError(
                f"🚫 Cannot create bucket '{bucket_name}'. ",
                f"Your service account needs 'roles/storage.admin' permission."
            )
    return bucket

def _normalize_folder(folder):
    """
    Normalize folder path for GCS upload
    """
    normalized_folder = None
    if folder:
        candidate = folder.strip("/").replace("\\", "/")
        normalized_folder = candidate or None
    return normalized_folder

def _truncate_name(blob_name, normalized_folder, separator="_____"):
    """
    Truncate the blob name to fit within GCS limits (1024 BYTES).
    """
    max_bytes = 1024
    # Calculate folder overhead in BYTES
    folder_overhead = len(normalized_folder.encode('utf-8')) + 1 if normalized_folder else 0
    available_bytes = max_bytes - folder_overhead
    separator_bytes = len(separator.encode('utf-8'))
    if available_bytes <= separator_bytes:
        raise ValueError("Folder path too long to create object name.")
    blob_name_bytes = blob_name.encode('utf-8')
    # If it already fits, return as-is
    if len(blob_name_bytes) <= available_bytes:
        return blob_name
    # Calculate suffix length in bytes
    desired_suffix_bytes = min(100, len(blob_name_bytes))
    max_suffix_bytes = available_bytes - separator_bytes - 1
    if max_suffix_bytes <= 0:
        raise ValueError("Folder path too long to create object name.")
    suffix_bytes = min(desired_suffix_bytes, max_suffix_bytes)
    base_bytes = available_bytes - separator_bytes - suffix_bytes
    if base_bytes <= 0:
        raise ValueError("Folder path too long to create object name.")
    # Truncate at byte boundaries and decode safely
    base_part = blob_name_bytes[:base_bytes].decode('utf-8', errors='ignore')
    suffix_part = blob_name_bytes[-suffix_bytes:].decode('utf-8', errors='ignore')
    truncated_name = f"{base_part}{separator}{suffix_part}"
    return truncated_name

def upload_to_storage(storage_client, bucket_name, pdf_obj, folder=None):
    """
    Upload the PDF object to Google Cloud Storage.
    If the bucket doesn't exist, try to create it (without triggering a 403 from bucket.exists()).
    Skip the upload when the document is already stored.
    Optionally upload inside a folder within the bucket.
    """
    bucket = _get_bucket(storage_client, bucket_name)
    content_type = CONTENT_TYPE_MAP.get(pdf_obj["extension"], "application/file")
    blob_name = pdf_obj["name"].replace("/", "_").replace("\\", "_")
    normalized_folder = _normalize_folder(folder)

    try:
        blob_path = f"{normalized_folder}/{blob_name}" if normalized_folder else blob_name
        if bucket.get_blob(blob_path) is not None:
            # print(f"ℹ️ Skipping upload; '{blob_path}' already exists in gs://{bucket_name}")
            return

        blob = bucket.blob(blob_path)
        
        blob.upload_from_string(pdf_obj["bytes"], content_type=content_type)
        print(f"✅ Uploaded '{blob_path}' to gs://{bucket_name}")
    except BadRequest as err:
        err_text = str(err)
        if ("maximum object length") in err_text or ("The bucket name and object name together must be at most 1087 characters") in err_text:
            truncated_name = _truncate_name(blob_name, normalized_folder, separator="_____")
            truncated_path = (
                f"{normalized_folder}/{truncated_name}" if normalized_folder else truncated_name
            )
            # print(f"⚠️ Blob name too long; retrying as '{truncated_path}'")
            # print("Len: " + str(len(truncated_path)))
            if bucket.get_blob(truncated_path) is not None:
                print(f"ℹ️ Skipping upload; '{truncated_path}' already exists in gs://{bucket_name}")
                return
            fallback_blob = bucket.blob(truncated_path)
            fallback_blob.upload_from_string(pdf_obj["bytes"], content_type=content_type)
            # print(f"✅ Uploaded '{truncated_path}' to gs://{bucket_name}")
        else:
            raise
