package expo.modules.geminanano

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.TextPart
import expo.modules.kotlin.functions.Coroutine
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.flow.first

class GeminiNanoModule : Module() {

    // Reuse across calls in the same app session to avoid AICore rebinding overhead.
    private val model by lazy { Generation.getClient() }

    override fun definition() = ModuleDefinition {
        Name("GeminiNano")

        // Returns: "available" | "downloading" | "downloadable" | "unavailable"
        // "downloadable" = device supports Gemini Nano, model not yet downloaded — call requestDownload()
        // "unavailable"  = device does not support Gemini Nano (no AICore / wrong device)
        AsyncFunction("checkAvailability") Coroutine { ->
            try {
                when (model.checkStatus()) {
                    FeatureStatus.AVAILABLE    -> "available"
                    FeatureStatus.DOWNLOADING  -> "downloading"
                    FeatureStatus.DOWNLOADABLE -> "downloadable"
                    else                       -> "unavailable"
                }
            } catch (e: Exception) {
                // Handles: unlocked bootloader, AICore not initialized, binding errors
                "unavailable"
            }
        }

        // Triggers AICore to download the Gemini Nano model.
        // Returns: "started" | "already_available" | "unavailable" | "error:<message>"
        AsyncFunction("requestDownload") Coroutine { ->
            try {
                when (model.checkStatus()) {
                    FeatureStatus.AVAILABLE -> "already_available"
                    FeatureStatus.UNAVAILABLE -> "unavailable"
                    else -> {
                        // Collect the first event from the download flow to kick it off.
                        // The flow emits DownloadStatus events; we return after the first one fires.
                        val firstStatus = model.download().first()
                        when (firstStatus) {
                            is DownloadStatus.DownloadStarted   -> "started"
                            is DownloadStatus.DownloadCompleted -> "already_available"
                            is DownloadStatus.DownloadFailed    -> "error:${firstStatus.e.message}"
                            else                                -> "started"
                        }
                    }
                }
            } catch (e: Exception) {
                "error:${e.message}"
            }
        }

        AsyncFunction("identifyFood") Coroutine { imageUri: String, prompt: String ->
            try {
                val androidUri = Uri.parse(imageUri)
                val stream = if (androidUri.scheme == "content") {
                    appContext.reactContext!!.contentResolver.openInputStream(androidUri)!!
                } else {
                    java.io.FileInputStream(androidUri.path!!)
                }

                // Decode bitmap and scale to max 1024px on longest edge to reduce latency.
                // Defensive against full 12MP camera photos causing excessive processing time.
                val rawBitmap = BitmapFactory.decodeStream(stream)
                val bitmap = scaleBitmapIfNeeded(rawBitmap, 1024)

                val requestBuilder = GenerateContentRequest.Builder(ImagePart(bitmap), TextPart(prompt))
                requestBuilder.temperature = 0.2f
                requestBuilder.maxOutputTokens = 256
                requestBuilder.topK = 10
                val request = requestBuilder.build()

                val response = model.generateContent(request)
                response.candidates.firstOrNull()?.text ?: ""
            } catch (e: Exception) {
                // Handles: BUSY quota error, stream errors, inference errors
                ""
            }
        }
    }

    /**
     * Scale bitmap down so the longest edge is at most maxPx.
     * Returns the original bitmap unchanged if already within bounds.
     */
    private fun scaleBitmapIfNeeded(bitmap: Bitmap, maxPx: Int): Bitmap {
        val maxEdge = maxOf(bitmap.width, bitmap.height)
        if (maxEdge <= maxPx) return bitmap
        val scale = maxPx.toFloat() / maxEdge
        val newWidth = (bitmap.width * scale).toInt()
        val newHeight = (bitmap.height * scale).toInt()
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
}
