/**
 * On-device text embedding service using MiniLM TFLite model.
 *
 * Loads the bundled embedding.tflite model via react-native-fast-tflite
 * and produces 384-dimensional Float32Array embeddings for food name
 * queries. Used by vlmPipeline for semantic USDA vector search alongside
 * BM25 keyword search.
 *
 * Lazy init per D-09: warmup() called on first detection flow, not at boot.
 */

import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { tokenize } from './wordpieceTokenizer';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const BUNDLED_MODEL = require('../../../assets/models/embedding.tflite');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const vocabJson = require('../../../assets/data/vocab_embedding.json');

const MAX_SEQ_LEN = 128;

export class EmbeddingService {
  private static instance: EmbeddingService | null = null;
  private model: TensorflowModel | null = null;
  private vocab: Map<string, number> | null = null;

  private constructor() {}

  static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  /** Reset singleton for testing. */
  static _resetForTesting(): void {
    EmbeddingService.instance = null;
  }

  async warmup(): Promise<void> {
    if (this.model) return; // idempotent
    this.model = await loadTensorflowModel(BUNDLED_MODEL, 'default');
    this.vocab = new Map(Object.entries(vocabJson as Record<string, number>));
  }

  async embed(text: string): Promise<Float32Array | null> {
    if (!this.model || !this.vocab) return null;
    const { inputIds, attentionMask } = tokenize(text, this.vocab, MAX_SEQ_LEN);
    const output = await this.model.run([inputIds, attentionMask]);
    const raw = output[0];
    return raw instanceof Float32Array
      ? raw
      : new Float32Array(raw as ArrayBuffer);
  }

  get ready(): boolean {
    return this.model !== null;
  }
}
