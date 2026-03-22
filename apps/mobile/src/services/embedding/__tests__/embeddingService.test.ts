jest.mock('react-native-fast-tflite', () => ({
  loadTensorflowModel: jest.fn().mockResolvedValue({
    run: jest.fn().mockResolvedValue([new Float32Array(384).fill(0.05)]),
  }),
}));

jest.mock('../../../../assets/data/vocab_embedding.json', () => ({
  '[PAD]': 0,
  '[UNK]': 100,
  '[CLS]': 101,
  '[SEP]': 102,
  chicken: 2000,
  breast: 2001,
}));

import { loadTensorflowModel } from 'react-native-fast-tflite';
import { EmbeddingService } from '../embeddingService';

describe('EmbeddingService', () => {
  beforeEach(() => {
    EmbeddingService._resetForTesting();
    (loadTensorflowModel as jest.Mock).mockClear();
  });

  it('returns the same instance (singleton)', () => {
    const a = EmbeddingService.getInstance();
    const b = EmbeddingService.getInstance();
    expect(a).toBe(b);
  });

  it('ready is false before warmup', () => {
    const svc = EmbeddingService.getInstance();
    expect(svc.ready).toBe(false);
  });

  it('embed() returns null before warmup', async () => {
    const svc = EmbeddingService.getInstance();
    const result = await svc.embed('chicken');
    expect(result).toBeNull();
  });

  it('ready is true after warmup', async () => {
    const svc = EmbeddingService.getInstance();
    await svc.warmup();
    expect(svc.ready).toBe(true);
  });

  it('embed() returns Float32Array of length 384 after warmup', async () => {
    const svc = EmbeddingService.getInstance();
    await svc.warmup();
    const result = await svc.embed('chicken');
    expect(result).toBeInstanceOf(Float32Array);
    expect(result!.length).toBe(384);
  });

  it('warmup() called twice loads model only once (idempotent)', async () => {
    const svc = EmbeddingService.getInstance();
    await svc.warmup();
    await svc.warmup();
    expect(loadTensorflowModel).toHaveBeenCalledTimes(1);
  });

  it('_resetForTesting() creates a fresh instance', () => {
    const a = EmbeddingService.getInstance();
    EmbeddingService._resetForTesting();
    const b = EmbeddingService.getInstance();
    expect(a).not.toBe(b);
  });
});
