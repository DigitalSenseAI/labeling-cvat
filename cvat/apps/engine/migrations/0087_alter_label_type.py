# Generated by Django 4.2.17 on 2025-01-22 13:48

from django.db import migrations, models

import cvat.apps.engine.models


class Migration(migrations.Migration):

    dependencies = [
        ("engine", "0086_profile_has_analytics_access"),
    ]

    operations = [
        migrations.AlterField(
            model_name="label",
            name="type",
            field=models.CharField(
                choices=[
                    ("any", "ANY"),
                    ("cuboid", "CUBOID"),
                    ("ellipse", "ELLIPSE"),
                    ("mask", "MASK"),
                    ("points", "POINTS"),
                    ("polygon", "POLYGON"),
                    ("polyline", "POLYLINE"),
                    ("rectangle", "RECTANGLE"),
                    ("skeleton", "SKELETON"),
                    ("tag", "TAG"),
                ],
                default=cvat.apps.engine.models.LabelType["ANY"],
                max_length=32,
            ),
        ),
    ]
