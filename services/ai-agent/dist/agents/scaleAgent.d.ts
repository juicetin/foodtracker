import { LlmAgent } from '@google/adk';
/**
 * ScaleAgent: Detects and reads weight from kitchen scales in images
 *
 * Responsibilities:
 * - Detect presence of kitchen scale
 * - Read digital scale display using OCR
 * - Extract weight value and unit
 * - Handle analog scales if possible
 */
export interface ScaleReading {
    detected: boolean;
    weight?: number;
    unit?: 'g' | 'kg' | 'oz' | 'lb';
    confidence: number;
    scaleType?: 'digital' | 'analog';
}
export declare const scaleAgent: LlmAgent;
