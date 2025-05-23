# Generated by Django 3.1.13 on 2021-08-27 02:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("engine", "0040_cloud_storage"),
    ]

    operations = [
        migrations.AlterField(
            model_name="cloudstorage",
            name="credentials_type",
            field=models.CharField(
                choices=[
                    ("TEMP_KEY_SECRET_KEY_TOKEN_SET", "TEMP_KEY_SECRET_KEY_TOKEN_SET"),
                    ("ACCOUNT_NAME_TOKEN_PAIR", "ACCOUNT_NAME_TOKEN_PAIR"),
                    ("KEY_FILE_PATH", "KEY_FILE_PATH"),
                    ("ANONYMOUS_ACCESS", "ANONYMOUS_ACCESS"),
                ],
                max_length=29,
            ),
        ),
        migrations.AlterField(
            model_name="cloudstorage",
            name="provider_type",
            field=models.CharField(
                choices=[
                    ("AWS_S3_BUCKET", "AWS_S3"),
                    ("AZURE_CONTAINER", "AZURE_CONTAINER"),
                    ("GOOGLE_DRIVE", "GOOGLE_DRIVE"),
                    ("GOOGLE_CLOUD_STORAGE", "GOOGLE_CLOUD_STORAGE"),
                ],
                max_length=20,
            ),
        ),
    ]
