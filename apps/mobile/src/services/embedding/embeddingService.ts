/**
 * On-device text embedding service — stub.
 *
 * The vec search infrastructure (usda_embeddings table, SQL_SEARCH_USDA_VEC) is
 * ready, but on-device query embedding needs a Metro-compatible runtime.
 *
 * Options under evaluation:
 *  A. react-native-fast-tflite + MiniLM TFLite export + WordPiece tokenizer in JS
 *  B. @huggingface/transformers once Metro ESM support lands in RN 0.82+
 *
 * Until one is wired up, embed() returns null so the vec path is skipped and
 * the pipeline falls through to BM25 → prefix → KG → proxy.
 */

export class EmbeddingService {
  private static instance: EmbeddingService | null = null;

  private constructor() {}

  static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  async warmup(): Promise<void> {
    // TODO: load model when runtime is available
  }

  async embed(_text: string): Promise<Float32Array | null> {
    // TODO: implement when Metro ESM / TFLite path is chosen
    return null;
  }

  get ready(): boolean {
    return false;
  }
}
