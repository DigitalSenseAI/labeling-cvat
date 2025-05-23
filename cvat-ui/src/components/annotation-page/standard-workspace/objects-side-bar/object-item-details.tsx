// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { Row, Col } from 'antd/lib/grid';
import Text from 'antd/lib/typography/Text';
import Collapse from 'antd/lib/collapse';
import InputNumber from 'antd/lib/input-number';

import ItemAttribute from './object-item-attribute';

interface Props {
    readonly: boolean;
    collapsed: boolean;
    attributes: any[];
    values: Record<number, string>;
    changeAttribute(attrID: number, value: string): void;
    collapse(): void;
    sizeParams: SizeParams | null;
    changeSize(sizeType: SizeType, value: number): void;
}

export enum SizeType {
    WIDTH = 'width',
    HEIGHT = 'height',
    LENGTH = 'length',
}

export interface SizeParams {
    width: number;
    height: number;
    length: number;
}

export function attrValuesAreEqual(next: Record<number, string>, prev: Record<number, string>): boolean {
    const prevKeys = Object.keys(prev);
    const nextKeys = Object.keys(next);

    return (
        nextKeys.length === prevKeys.length &&
        nextKeys.map((key: string): boolean => prev[+key] === next[+key]).every((value: boolean) => value)
    );
}

function attrAreTheSame(prevProps: Props, nextProps: Props): boolean {
    return (
        nextProps.readonly === prevProps.readonly &&
        nextProps.collapsed === prevProps.collapsed &&
        nextProps.attributes === prevProps.attributes &&
        attrValuesAreEqual(nextProps.values, prevProps.values)
    );
}

function ItemAttributesComponent(props: Props): JSX.Element {
    const {
        collapsed, attributes, values, readonly, changeAttribute, collapse,
        sizeParams, changeSize,
    } = props;

    return (
        <Row>
            <Collapse
                className='cvat-objects-sidebar-state-item-collapse'
                activeKey={collapsed ? [] : ['details']}
                onChange={collapse}
                items={[{
                    key: 'details',
                    label: <Text style={{ fontSize: 10 }} type='secondary'>DETAILS</Text>,
                    children: [
                        sizeParams && (
                            <Row key='size' justify='space-around' className='cvat-objects-sidebar-size-params'>
                                {Object.keys(sizeParams).map((key) => (
                                    <Col key={key}>
                                        <Text type='secondary'>
                                            {`${key.charAt(0).toUpperCase()}:`}
                                        </Text>
                                        <InputNumber
                                            value={sizeParams[key as keyof SizeParams] || ''}
                                            onChange={(value) => {
                                                if (typeof value === 'number') {
                                                    changeSize(
                                                        SizeType[key.toUpperCase() as keyof typeof SizeType],
                                                        value,
                                                    );
                                                }
                                            }}
                                            disabled={readonly}
                                        />
                                    </Col>
                                ))}
                            </Row>
                        ),
                        ...attributes.map(
                            (attribute: any): JSX.Element => (
                                <Row
                                    key={attribute.id}
                                    align='middle'
                                    justify='start'
                                    className='cvat-object-item-attribute-wrapper'
                                >
                                    <ItemAttribute
                                        readonly={readonly}
                                        attrValue={values[attribute.id]}
                                        attrInputType={attribute.inputType}
                                        attrName={attribute.name}
                                        attrID={attribute.id}
                                        attrValues={attribute.values}
                                        changeAttribute={changeAttribute}
                                    />
                                </Row>
                            ),
                        )],
                }]}
            />
        </Row>
    );
}

export default React.memo(ItemAttributesComponent, attrAreTheSame);
