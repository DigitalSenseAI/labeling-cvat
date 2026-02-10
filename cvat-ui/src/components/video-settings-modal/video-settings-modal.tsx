// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import './styles.scss';

import React, { useState, useEffect, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import Modal from 'antd/lib/modal';
import { Row, Col } from 'antd/lib/grid';
import Form from 'antd/lib/form';
import Input from 'antd/lib/input';
import Switch from 'antd/lib/switch';
import Button from 'antd/lib/button';
import Typography from 'antd/lib/typography';
import notification from 'antd/lib/notification';
import Spin from 'antd/lib/spin';
import {
    EditOutlined,
    InfoCircleOutlined,
    WarningOutlined,
    ExclamationCircleOutlined,
} from '@ant-design/icons';

import { Task, getCore } from 'cvat-core-wrapper';
import CVATTooltip from 'components/common/cvat-tooltip';

const { Text, Title } = Typography;
const core = getCore();

interface VideoSettings {
    useZipChunks: boolean;
    useCache: boolean;
    imageQuality: number;
    chunkSize: number;
    originalChunkQuality: number;
    activeJobsUsers: string[];
}

interface Props {
    visible: boolean;
    taskInstance: Task;
    onClose: () => void;
    onSettingsUpdated?: () => void;
}

function VideoSettingsModal({
    visible,
    taskInstance,
    onClose,
    onSettingsUpdated,
}: Props): JSX.Element {
    const dispatch = useDispatch();
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [settings, setSettings] = useState<VideoSettings | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Edit states for each field
    const [editingField, setEditingField] = useState<string | null>(null);
    const [editValues, setEditValues] = useState<Partial<VideoSettings>>({});

    const fetchSettings = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const fetchedSettings = await core.server.request(
                `${core.config.backendAPI}/tasks/${taskInstance.id}/video-settings`,
                { method: 'GET' },
            );
            setSettings({
                useZipChunks: fetchedSettings.use_zip_chunks,
                useCache: fetchedSettings.use_cache,
                imageQuality: fetchedSettings.image_quality,
                chunkSize: fetchedSettings.chunk_size,
                originalChunkQuality: fetchedSettings.original_chunk_quality,
                activeJobsUsers: fetchedSettings.active_jobs_users || [],
            });
        } catch (err: any) {
            setError(err.message || 'Failed to fetch video settings');
        } finally {
            setLoading(false);
        }
    }, [taskInstance.id]);

    useEffect(() => {
        if (visible) {
            fetchSettings();
            setEditingField(null);
            setEditValues({});
        }
    }, [visible, fetchSettings]);

    const handleEdit = (field: string) => {
        if (!settings) return;
        setEditingField(field);
        setEditValues({
            ...editValues,
            [field]: settings[field as keyof VideoSettings],
        });
    };

    const handleCancelEdit = () => {
        setEditingField(null);
    };

    const handleSaveEdit = (field: string, value: any) => {
        setEditValues({
            ...editValues,
            [field]: value,
        });
        setEditingField(null);
    };

    const hasChanges = (): boolean => {
        if (!settings) return false;
        return Object.keys(editValues).some(
            (key) => editValues[key as keyof VideoSettings] !== settings[key as keyof VideoSettings],
        );
    };

    const handleCancel = () => {
        setEditingField(null);
        setEditValues({});
        onClose();
    };

    const handleConfirm = async () => {
        if (!settings || !hasChanges()) {
            onClose();
            return;
        }

        // Check if other users are working on the task
        if (settings.activeJobsUsers.length > 0) {
            notification.error({
                message: 'Cannot update video settings',
                description: `Other users are currently working on this task: ${settings.activeJobsUsers.join(', ')}`,
            });
            return;
        }

        // Build confirmation message
        const changes: string[] = [];
        if (editValues.useZipChunks !== undefined && editValues.useZipChunks !== settings.useZipChunks) {
            changes.push(`Prefer zip chunks: ${settings.useZipChunks} → ${editValues.useZipChunks}`);
        }
        if (editValues.useCache !== undefined && editValues.useCache !== settings.useCache) {
            changes.push(`Use cache: ${settings.useCache} → ${editValues.useCache}`);
        }
        if (editValues.imageQuality !== undefined && editValues.imageQuality !== settings.imageQuality) {
            changes.push(`Image quality: ${settings.imageQuality}% → ${editValues.imageQuality}%`);
        }
        if (editValues.chunkSize !== undefined && editValues.chunkSize !== settings.chunkSize) {
            changes.push(`Chunk size: ${settings.chunkSize} → ${editValues.chunkSize}`);
        }

        // Determine if quality warning is needed
        const showQualityWarning = settings.originalChunkQuality === 67 && (
            (editValues.imageQuality !== undefined && editValues.imageQuality > 67) ||
            (editValues.useZipChunks !== undefined && editValues.useZipChunks !== settings.useZipChunks) ||
            (editValues.chunkSize !== undefined && editValues.chunkSize !== settings.chunkSize)
        );

        Modal.confirm({
            title: 'Are you sure you want to change the current video settings?',
            icon: <ExclamationCircleOutlined />,
            className: 'cvat-video-settings-confirm-modal',
            content: (
                <div>
                    <p>The following settings will be changed:</p>
                    <ul>
                        {changes.map((change, idx) => (
                            <li key={idx}>{change}</li>
                        ))}
                    </ul>
                    <p>
                        <WarningOutlined style={{ color: '#faad14' }} />
                        {' '}
                        The task will be temporarily unavailable while chunks are being regenerated.
                    </p>
                    {showQualityWarning && (
                        <div className="cvat-video-settings-quality-warning">
                            <WarningOutlined style={{ color: '#faad14' }} />
                            {' '}
                            <strong>Note:</strong>
                            {' '}
                            The original video chunks were stored at quality 67 (visually lossless).
                            The new compressed chunks will be generated from these, so the maximum
                            achievable quality is 67.
                        </div>
                    )}
                </div>
            ),
            okText: 'Confirm',
            cancelText: 'Cancel',
            onOk: async () => {
                setSubmitting(true);
                try {
                    // Build request body with only changed values
                    const body: Record<string, any> = {};
                    if (editValues.useZipChunks !== undefined && editValues.useZipChunks !== settings.useZipChunks) {
                        body.use_zip_chunks = editValues.useZipChunks;
                    }
                    if (editValues.useCache !== undefined && editValues.useCache !== settings.useCache) {
                        body.use_cache = editValues.useCache;
                    }
                    if (editValues.imageQuality !== undefined && editValues.imageQuality !== settings.imageQuality) {
                        body.image_quality = editValues.imageQuality;
                    }
                    if (editValues.chunkSize !== undefined && editValues.chunkSize !== settings.chunkSize) {
                        body.chunk_size = editValues.chunkSize;
                    }

                    await core.server.request(
                        `${core.config.backendAPI}/tasks/${taskInstance.id}/video-settings`,
                        {
                            method: 'PATCH',
                            data: body,
                        },
                    );

                    notification.success({
                        message: 'Video settings update started',
                        description: 'The task chunks are being regenerated. This may take some time.',
                    });

                    if (onSettingsUpdated) {
                        onSettingsUpdated();
                    }
                    handleCancel();
                } catch (err: any) {
                    if (err.response?.status === 409) {
                        notification.error({
                            message: 'Cannot update video settings',
                            description: `Other users are currently working on this task: ${err.response.data.active_users?.join(', ') || 'Unknown'}`,
                        });
                    } else {
                        notification.error({
                            message: 'Failed to update video settings',
                            description: err.message || 'Unknown error',
                        });
                    }
                } finally {
                    setSubmitting(false);
                }
            },
        });
    };

    const renderField = (
        label: string,
        field: keyof VideoSettings,
        tooltip: string,
        renderValue: (value: any) => React.ReactNode,
        renderEditor: (value: any, onChange: (val: any) => void) => React.ReactNode,
    ) => {
        if (!settings) return null;

        const currentValue = editValues[field] !== undefined ? editValues[field] : settings[field];
        const isEditing = editingField === field;
        const hasBeenEdited = editValues[field] !== undefined && editValues[field] !== settings[field];

        return (
            <div className="cvat-video-settings-form-item">
                <div className="cvat-video-settings-field-label">
                    <Text strong>{label}</Text>
                    <CVATTooltip title={tooltip}>
                        <InfoCircleOutlined style={{ color: '#8c8c8c' }} />
                    </CVATTooltip>
                    {hasBeenEdited && (
                        <Text type="warning" style={{ fontSize: 12 }}>
                            (modified)
                        </Text>
                    )}
                </div>
                <div className="cvat-video-settings-field-value">
                    {isEditing ? (
                        <>
                            <div className="cvat-video-settings-field-input">
                                {renderEditor(currentValue, (val) => handleSaveEdit(field, val))}
                            </div>
                            <Button size="small" onClick={handleCancelEdit}>
                                Cancel
                            </Button>
                        </>
                    ) : (
                        <>
                            <div className="cvat-video-settings-field-input">
                                {renderValue(currentValue)}
                            </div>
                            <EditOutlined
                                className="cvat-video-settings-edit-btn"
                                onClick={() => handleEdit(field)}
                            />
                        </>
                    )}
                </div>
            </div>
        );
    };

    return (
        <Modal
            title="Video Settings"
            open={visible}
            onCancel={handleCancel}
            className="cvat-video-settings-modal"
            footer={[
                <Button key="cancel" onClick={handleCancel}>
                    Cancel
                </Button>,
                <Button
                    key="confirm"
                    type="primary"
                    onClick={handleConfirm}
                    disabled={!hasChanges() || submitting || (settings?.activeJobsUsers.length ?? 0) > 0}
                    loading={submitting}
                >
                    Confirm
                </Button>,
            ]}
            width={500}
        >
            {loading ? (
                <div style={{ textAlign: 'center', padding: 40 }}>
                    <Spin size="large" />
                </div>
            ) : error ? (
                <div style={{ textAlign: 'center', padding: 40 }}>
                    <Text type="danger">{error}</Text>
                </div>
            ) : settings ? (
                <>
                    {settings.activeJobsUsers && settings.activeJobsUsers.length > 0 && (
                        <div className="cvat-video-settings-active-users">
                            <WarningOutlined />
                            <Text strong>Cannot modify settings</Text>
                            <p>
                                The following users are currently working on this task
                                (jobs in progress):
                            </p>
                            <ul>
                                {settings.activeJobsUsers.map((user) => (
                                    <li key={user}>{user}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {renderField(
                        'Prefer zip chunks',
                        'useZipChunks',
                        'ZIP chunks have better quality, but they require more disk space and time to download',
                        (value) => (
                            <Text>{value ? 'Yes' : 'No'}</Text>
                        ),
                        (value, onChange) => (
                            <Switch
                                checked={value}
                                onChange={onChange}
                            />
                        ),
                    )}

                    {renderField(
                        'Use cache',
                        'useCache',
                        'Enable or disable task data chunk caching',
                        (value) => (
                            <Text>{value ? 'Enabled' : 'Disabled'}</Text>
                        ),
                        (value, onChange) => (
                            <Switch
                                checked={value}
                                onChange={onChange}
                            />
                        ),
                    )}

                    {renderField(
                        'Image quality',
                        'imageQuality',
                        `Quality of compressed chunks (5-100). Original chunks quality: ${settings.originalChunkQuality}`,
                        (value) => (
                            <Text>{value}%</Text>
                        ),
                        (value, onChange) => (
                            <Input
                                type="number"
                                min={5}
                                max={100}
                                defaultValue={value}
                                onBlur={(e) => {
                                    const numVal = parseInt(e.target.value, 10);
                                    if (numVal >= 5 && numVal <= 100) {
                                        onChange(numVal);
                                    }
                                }}
                                onPressEnter={(e) => {
                                    const numVal = parseInt((e.target as HTMLInputElement).value, 10);
                                    if (numVal >= 5 && numVal <= 100) {
                                        onChange(numVal);
                                    }
                                }}
                                suffix="%"
                                style={{ width: 100 }}
                            />
                        ),
                    )}

                    {renderField(
                        'Chunk size',
                        'chunkSize',
                        'Number of frames per chunk. Recommended: 36 for 1080p, 8-16 for 2K, 4-8 for 4K',
                        (value) => (
                            <Text>{value} frames</Text>
                        ),
                        (value, onChange) => (
                            <Input
                                type="number"
                                min={1}
                                defaultValue={value}
                                onBlur={(e) => {
                                    const numVal = parseInt(e.target.value, 10);
                                    if (numVal >= 1) {
                                        onChange(numVal);
                                    }
                                }}
                                onPressEnter={(e) => {
                                    const numVal = parseInt((e.target as HTMLInputElement).value, 10);
                                    if (numVal >= 1) {
                                        onChange(numVal);
                                    }
                                }}
                                suffix="frames"
                                style={{ width: 120 }}
                            />
                        ),
                    )}

                    {settings.originalChunkQuality === 67 && (
                        <div className="cvat-video-settings-info">
                            <InfoCircleOutlined />
                            <Text>
                                Original chunks are stored at quality 67 (visually lossless).
                                When regenerating, the maximum quality achievable is 67.
                            </Text>
                        </div>
                    )}
                </>
            ) : null}
        </Modal>
    );
}

export default VideoSettingsModal;
