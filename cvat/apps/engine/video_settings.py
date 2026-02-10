# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Module for handling video task settings changes and chunk regeneration.
"""

import os
import shutil
from typing import Any

import django_rq
import rq
from django.conf import settings
from django.db import transaction

from cvat.apps.engine import models
from cvat.apps.engine.frame_provider import FrameQuality, SegmentFrameProvider
from cvat.apps.engine.log import ServerLogManager
from cvat.apps.engine.media_extractors import (
    Mpeg4CompressedChunkWriter,
    ZipCompressedChunkWriter,
)
from cvat.apps.engine.models import RequestAction, RequestTarget
from cvat.apps.engine.rq import ImportRQMeta, RQId, define_dependent_job
from cvat.apps.engine.types import ExtendedRequest
from cvat.apps.engine.utils import get_rq_lock_by_user

slogger = ServerLogManager(__name__)


class RequestSubresourceVideoSettings:
    """Custom subresource for video settings regeneration requests."""
    VIDEO_SETTINGS = "video_settings"


def regenerate_chunks(
    db_task: models.Task,
    settings_data: dict[str, Any],
    request: ExtendedRequest,
) -> str:
    """
    Schedule a background job to regenerate video chunks with new settings.

    Args:
        db_task: The task to regenerate chunks for
        settings_data: Dictionary containing the new settings (use_zip_chunks, use_cache,
                      image_quality, chunk_size)
        request: The HTTP request that triggered this

    Returns:
        The RQ job ID for tracking the regeneration process
    """
    q = django_rq.get_queue(settings.CVAT_QUEUES.IMPORT_DATA.value)
    user_id = request.user.id

    # Create a unique RQ ID for this operation
    rq_id = RQId(
        RequestAction.CREATE,
        RequestTarget.TASK,
        db_task.pk,
    ).render() + f":video_settings"

    with get_rq_lock_by_user(q, user_id):
        q.enqueue_call(
            func=_regenerate_chunks_thread,
            args=(db_task.pk, settings_data),
            job_id=rq_id,
            meta=ImportRQMeta.build_for(request=request, db_obj=db_task),
            depends_on=define_dependent_job(q, user_id),
            failure_ttl=settings.IMPORT_CACHE_FAILED_TTL.total_seconds(),
        )

    return rq_id


def _regenerate_chunks_thread(
    task_id: int,
    settings_data: dict[str, Any],
) -> None:
    """
    Background thread for regenerating chunks.

    This function:
    1. Reads frames from original chunks (high quality source)
    2. Deletes existing compressed chunks
    3. Regenerates compressed chunks with new settings
    4. Updates the database with new settings
    """
    db_task = models.Task.objects.select_for_update().get(pk=task_id)
    db_data = db_task.data

    job = rq.get_current_job()
    rq_job_meta = ImportRQMeta.for_job(job)

    def update_status(msg: str, progress: float = None) -> None:
        rq_job_meta.status = msg
        if progress is not None:
            rq_job_meta.task_progress = progress
        rq_job_meta.save()

    slogger.glob.info(f"Starting chunk regeneration for task #{task_id}")
    update_status("Starting chunk regeneration...", 0)

    # Extract new settings
    new_use_zip_chunks = settings_data.get('use_zip_chunks')
    new_use_cache = settings_data.get('use_cache')
    new_image_quality = settings_data.get('image_quality')
    new_chunk_size = settings_data.get('chunk_size')

    # Determine if we need to regenerate chunks
    # (any change except use_cache requires regeneration)
    need_regeneration = any([
        new_use_zip_chunks is not None and (
            (new_use_zip_chunks and db_data.compressed_chunk_type != models.DataChoice.IMAGESET) or
            (not new_use_zip_chunks and db_data.compressed_chunk_type != models.DataChoice.VIDEO)
        ),
        new_image_quality is not None and new_image_quality != db_data.image_quality,
        new_chunk_size is not None and new_chunk_size != db_data.chunk_size,
    ])

    if need_regeneration:
        _do_regenerate_chunks(
            db_task=db_task,
            db_data=db_data,
            new_use_zip_chunks=new_use_zip_chunks,
            new_image_quality=new_image_quality,
            new_chunk_size=new_chunk_size,
            update_status=update_status,
        )

    # Update database settings
    with transaction.atomic():
        db_data = models.Data.objects.select_for_update().get(pk=db_data.pk)

        if new_use_zip_chunks is not None:
            db_data.compressed_chunk_type = (
                models.DataChoice.IMAGESET if new_use_zip_chunks
                else models.DataChoice.VIDEO
            )

        if new_use_cache is not None:
            db_data.storage_method = (
                models.StorageMethodChoice.CACHE if new_use_cache
                else models.StorageMethodChoice.FILE_SYSTEM
            )

        if new_image_quality is not None:
            db_data.image_quality = new_image_quality

        if new_chunk_size is not None:
            db_data.chunk_size = new_chunk_size

        db_data.save()

    update_status("Chunk regeneration completed", 1.0)
    slogger.glob.info(f"Chunk regeneration completed for task #{task_id}")


def _do_regenerate_chunks(
    db_task: models.Task,
    db_data: models.Data,
    new_use_zip_chunks: bool | None,
    new_image_quality: int | None,
    new_chunk_size: int | None,
    update_status,
) -> None:
    """
    Actually regenerate the compressed chunks from original chunks.
    """
    # Determine final settings
    use_zip = new_use_zip_chunks if new_use_zip_chunks is not None else (
        db_data.compressed_chunk_type == models.DataChoice.IMAGESET
    )
    quality = new_image_quality if new_image_quality is not None else db_data.image_quality
    chunk_size = new_chunk_size if new_chunk_size is not None else db_data.chunk_size

    # Select chunk writer based on format
    if use_zip:
        chunk_writer_class = ZipCompressedChunkWriter
    else:
        chunk_writer_class = Mpeg4CompressedChunkWriter

    chunk_writer_kwargs = {}
    if db_task.dimension == models.DimensionType.DIM_3D:
        chunk_writer_kwargs["dimension"] = db_task.dimension

    chunk_writer = chunk_writer_class(quality, **chunk_writer_kwargs)

    # Get all segments
    db_segments = list(db_task.segment_set.order_by('start_frame').all())
    total_segments = len(db_segments)

    update_status("Preparing to regenerate chunks...", 0.05)

    # Delete existing compressed chunks directory content
    compressed_dir = db_data.get_compressed_cache_dirname()
    if os.path.exists(compressed_dir):
        for filename in os.listdir(compressed_dir):
            filepath = os.path.join(compressed_dir, filename)
            try:
                os.remove(filepath)
            except Exception as e:
                slogger.glob.warning(f"Failed to remove {filepath}: {e}")
    else:
        os.makedirs(compressed_dir)

    # Process each segment
    for segment_idx, db_segment in enumerate(db_segments):
        segment_progress = segment_idx / total_segments
        update_status(
            f"Processing segment {segment_idx + 1}/{total_segments}...",
            0.1 + segment_progress * 0.85
        )

        _regenerate_segment_chunks(
            db_segment=db_segment,
            db_data=db_data,
            chunk_writer=chunk_writer,
            chunk_size=chunk_size,
        )

    update_status("Finalizing...", 0.98)


def _regenerate_segment_chunks(
    db_segment: models.Segment,
    db_data: models.Data,
    chunk_writer,
    chunk_size: int,
) -> None:
    """
    Regenerate chunks for a single segment using frames from original chunks.
    """
    # Create frame provider to read from original chunks
    frame_provider = SegmentFrameProvider(db_segment)

    # Get all frame numbers in this segment
    segment_frames = sorted(db_segment.frame_set)

    if not segment_frames:
        return

    # Group frames into chunks
    chunk_idx = 0
    frame_buffer = []

    for frame_num in segment_frames:
        # Get frame from original quality chunks
        frame_data = frame_provider.get_frame(
            frame_num,
            quality=FrameQuality.ORIGINAL,
        )
        frame_buffer.append(frame_data[0])  # frame_data is (frame, filename, _)

        # When buffer is full, write a chunk
        if len(frame_buffer) >= chunk_size:
            _write_chunk(
                db_data=db_data,
                db_segment=db_segment,
                chunk_writer=chunk_writer,
                chunk_idx=chunk_idx,
                frames=frame_buffer,
            )
            chunk_idx += 1
            frame_buffer = []

    # Write remaining frames
    if frame_buffer:
        _write_chunk(
            db_data=db_data,
            db_segment=db_segment,
            chunk_writer=chunk_writer,
            chunk_idx=chunk_idx,
            frames=frame_buffer,
        )

    # Clean up
    frame_provider.unload()


def _write_chunk(
    db_data: models.Data,
    db_segment: models.Segment,
    chunk_writer,
    chunk_idx: int,
    frames: list,
) -> None:
    """Write a single chunk to disk."""
    # Determine chunk type for path generation
    if isinstance(chunk_writer, ZipCompressedChunkWriter):
        chunk_type = models.DataChoice.IMAGESET
    else:
        chunk_type = models.DataChoice.VIDEO

    # Build chunk path (we need to handle this manually since we're changing the type)
    ext = 'zip' if chunk_type == models.DataChoice.IMAGESET else 'mp4'
    chunk_name = f'segment_{db_segment.id}-{chunk_idx}.{ext}'
    chunk_path = os.path.join(db_data.get_compressed_cache_dirname(), chunk_name)

    chunk_writer.save_as_chunk(
        images=frames,
        chunk_path=chunk_path,
    )
