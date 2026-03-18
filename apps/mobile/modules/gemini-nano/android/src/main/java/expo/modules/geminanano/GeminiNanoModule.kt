package expo.modules.geminanano

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.google.mlkit.genai.prompt.FeatureStatus
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.TextPart
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class GeminiNanoModule : Module() {

    // Reuse across calls in the same app session to avoid AICore rebinding overhead.
    private val model by lazy { Generation.getClient() }

    override fun definition() = ModuleDefinition {
        Name("GeminiNano")

        AsyncFunction("checkAvailability") Coroutine { ->
            try {
                when (model.checkStatus()) {
                    FeatureStatus.AVAILABLE -> "available"
                    FeatureStatus.DOWNLOADING -> "downloading"
                    else -> "not_supported"
                }
            } catch (e: Exception) {
                // Handles: unlocked bootloader, AICore not initialized, binding errors
                "not_supported"
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

                val request = GenerateContentRequest
                    .builder(ImagePart(bitmap), TextPart(prompt))
                    .setTemperature(0.2f)
                    .setMaxOutputTokens(256)
                    .setTopK(10)
                    .build()

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
