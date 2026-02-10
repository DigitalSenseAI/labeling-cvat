// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Col } from 'antd/lib/grid';
import Select from 'antd/lib/select';
import Spin from 'antd/lib/spin';
import notification from 'antd/lib/notification';
import Icon from '@ant-design/icons';
import { TagsOutlined } from '@ant-design/icons';
import { CombinedState } from 'reducers';
import { changeFrameAsync } from 'actions/annotation-actions';

interface LabelFrameData {
    labelName: string;
    frames: number[];
}

function LabelsNavigationMenu(): JSX.Element | null {
    const dispatch = useDispatch();
    const [labelFrames, setLabelFrames] = useState<LabelFrameData[]>([]);
    const [loading, setLoading] = useState(false);
    const [selectedLabel, setSelectedLabel] = useState<string | undefined>(undefined);

    const {
        jobInstance,
        frameNumber,
        labels,
    } = useSelector((state: CombinedState) => ({
        jobInstance: state.annotation.job.instance,
        frameNumber: state.annotation.player.frame.number,
        labels: state.annotation.job.labels,
    }));

    const loadLabelFrames = useCallback(async () => {
        if (!jobInstance) return;

        setLoading(true);
        try {
            const labelsData: Record<string, Set<number>> = {};

            // Initialize all labels
            labels.forEach((label: any) => {
                labelsData[label.name] = new Set<number>();
            });

            // Get all frame numbers for the job
            const { startFrame, stopFrame } = jobInstance;

            // Use a more efficient approach: search for frames with annotations
            // This is much faster than loading all frames
            try {
                const statistics = await jobInstance.annotations.statistics();

                // If there are no annotations, return early
                if (!statistics || !statistics.total ||
                    !Object.values(statistics.total).some((val: any) =>
                        typeof val === 'object' ? Object.values(val).some((v: any) => v > 0) : val > 0
                    )) {
                    setLabelFrames([]);
                    return;
                }

                // Sample frames to find which ones have annotations
                // We'll check every 10th frame first, then fill in gaps
                const framesToCheck: number[] = [];
                const step = Math.max(1, Math.floor((stopFrame - startFrame + 1) / 50)); // Check max 50 frames

                for (let frame = startFrame; frame <= stopFrame; frame += step) {
                    framesToCheck.push(frame);
                }

                // Also add the last frame if not included
                if (framesToCheck[framesToCheck.length - 1] !== stopFrame) {
                    framesToCheck.push(stopFrame);
                }

                // Load annotations for sampled frames
                for (const frame of framesToCheck) {
                    try {
                        const annotations = await jobInstance.annotations.get(frame, false, []);

                        if (annotations && annotations.length > 0) {
                            annotations.forEach((annotation: any) => {
                                const labelName = annotation.label.name;
                                if (labelsData[labelName]) {
                                    labelsData[labelName].add(frame);
                                }
                            });
                        }
                    } catch (error) {
                        // Skip frames with errors
                        console.warn(`Error loading annotations for frame ${frame}:`, error);
                    }
                }
            } catch (error) {
                console.warn('Error loading statistics:', error);
            }

            // Convert to array format
            const result: LabelFrameData[] = Object.entries(labelsData)
                .filter(([, frames]) => frames.size > 0)
                .map(([labelName, frames]) => ({
                    labelName,
                    frames: Array.from(frames).sort((a, b) => a - b),
                }));

            setLabelFrames(result);
        } catch (error) {
            notification.error({
                message: 'Could not load label frames',
                description: error instanceof Error ? error.message : 'Unknown error',
            });
        } finally {
            setLoading(false);
        }
    }, [jobInstance, labels]);

    useEffect(() => {
        if (jobInstance && labels.length > 0) {
            loadLabelFrames();
        }
    }, [jobInstance?.id]);

    const handleLabelChange = useCallback((value: string) => {
        setSelectedLabel(value);
    }, []);

    const handleFrameSelect = useCallback((frame: number) => {
        dispatch(changeFrameAsync(frame));
    }, [dispatch]);

    if (!jobInstance || labelFrames.length === 0) {
        return null;
    }

    const selectedLabelData = labelFrames.find((lf: LabelFrameData) => lf.labelName === selectedLabel);

    return (
        <Col className='cvat-annotation-header-labels-navigation'>
            <Select
                style={{ width: 200 }}
                placeholder={
                    <>
                        <Icon component={TagsOutlined} style={{ marginRight: 4 }} />
                        Navigate by labels
                    </>
                }
                value={selectedLabel}
                onChange={handleLabelChange}
                loading={loading}
                allowClear
                onClear={() => setSelectedLabel(undefined)}
                dropdownMatchSelectWidth={false}
            >
                {labelFrames.map((labelFrame: LabelFrameData) => (
                    <Select.Option key={labelFrame.labelName} value={labelFrame.labelName}>
                        {`${labelFrame.labelName} (${labelFrame.frames.length} frames)`}
                    </Select.Option>
                ))}
            </Select>
            {selectedLabelData && (
                <Select
                    style={{ width: 150, marginLeft: 8 }}
                    placeholder="Select frame"
                    value={frameNumber}
                    onChange={handleFrameSelect}
                    dropdownMatchSelectWidth={false}
                >
                    {selectedLabelData.frames.map((frame: number) => (
                        <Select.Option key={frame} value={frame}>
                            Frame {frame}
                        </Select.Option>
                    ))}
                </Select>
            )}
            {loading && <Spin size="small" style={{ marginLeft: 8 }} />}
        </Col>
    );
}

export default React.memo(LabelsNavigationMenu);
