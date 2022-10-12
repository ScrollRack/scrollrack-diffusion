import boto3 
import uuid
import os
import io

def send(image):
    filename = str(uuid.uuid4()) + '.jpg'
    mem_file = io.BytesIO()
    image.save(mem_file, format='JPEG')
    mem_file.seek(0)
    
    s3 = boto3.client(
        service_name='s3',
        endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )

    bucket = f"{os.environ.get('S3_BUCKET')}"
    image_root_url = f"{os.environ.get('IMAGE_ROOT_URL')}"

    s3.upload_fileobj(
        mem_file,
        Bucket=bucket,
        Key=filename,
        ExtraArgs={
            'ContentType': 'image/jpeg',
        }
    )

    return f"{image_root_url}{filename}"