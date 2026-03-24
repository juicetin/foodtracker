package expo.modules.geminanano

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.PromptPrefix
import com.google.mlkit.genai.prompt.TextPart
import expo.modules.kotlin.functions.Coroutine
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext

class GeminiNanoModule : Module() {

    // Initialized eagerly on the main thread (module construction happens on main thread in Expo).
    // lazy { } would defer init to the first background coroutine caller — AICore binding
    // appears to require a main-looper context.
    private val model = Generation.getClient()

    override fun definition() = ModuleDefinition {
        Name("GeminiNano")

        // Returns: "available" | "downloading" | "downloadable" | "unavailable" | "needs_update"
        // "needs_update"  = AICore version too old; RC01 (versionCode < 382178) has an inference NPE bug
        // "downloadable"  = device supports Gemini Nano, model not yet downloaded — call requestDownload()
        // "unavailable"   = device does not support Gemini Nano (no AICore / wrong device)
        AsyncFunction("checkAvailability") Coroutine { ->
            try {
                val aiCoreVersion = try {
                    appContext.reactContext!!.packageManager
                        .getPackageInfo("com.google.android.aicore", 0).longVersionCode
                } catch (e: Exception) { -1L }

                // RC01 (382100) has a NullPointerException in AiCoreIsolatedService that causes
                // all generateContent() calls to fail with INFERENCE_ERROR for third-party apps.
                // RC02 (382178) fixes this. Minimum known-working versionCode: 382178.
                if (aiCoreVersion in 1 until 382178) return@Coroutine "needs_update"

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

        // Diagnostic probe — reports warmup result, model name, token limit, then tries inference.
        AsyncFunction("testTextOnly") Coroutine { prompt: String ->
            val sb = StringBuilder()
            try {
                try {
                    val name = model.getBaseModelName()
                    sb.append("model:$name ")
                } catch (e: Exception) {
                    sb.append("model:ERR:${e.message} ")
                }
                try {
                    val limit = model.getTokenLimit()
                    sb.append("tokenLimit:$limit ")
                } catch (e: Exception) {
                    sb.append("tokenLimit:ERR ")
                }
                // Use the simple String overload — bypass GenerateContentRequest entirely.
                // The Builder path triggers a NullPointerException inside AICore's isolated service.
                val response = withContext(Dispatchers.Main) { model.generateContent(prompt) }
                val text = response.candidates.firstOrNull()?.text
                sb.append("result:${text ?: "empty"}")
            } catch (e: com.google.mlkit.genai.common.GenAiException) {
                sb.append("infer:FAIL:code=${e.errorCode}:${e.message}")
            } catch (e: Exception) {
                sb.append("infer:FAIL:${e.javaClass.simpleName}:${e.message}")
            }
            sb.toString()
        }

        AsyncFunction("identifyFood") Coroutine { imageUri: String, prompt: String ->
            try {
                val androidUri = Uri.parse(imageUri)
                val stream = if (androidUri.scheme == "content") {
                    appContext.reactContext!!.contentResolver.openInputStream(androidUri)!!
                } else {
                    java.io.FileInputStream(androidUri.path!!)
                }

                // Decode bitmap as ARGB_8888 — BitmapFactory returns HARDWARE config by default
                // on Android 8+ which is GPU-only and unreadable by ML Kit CPU inference.
                val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
                val rawBitmap = BitmapFactory.decodeStream(stream, null, opts)
                    ?: return@Coroutine "ERROR:decode_failed:null bitmap from stream"
                val bitmap = scaleBitmapIfNeeded(rawBitmap, 512)

                val requestBuilder = GenerateContentRequest.Builder(ImagePart(bitmap), TextPart(prompt))
                requestBuilder.temperature = 0.2f
                // ML Kit Prompt API may hard-limit maxOutputTokens to 256 at runtime.
                // We request 1024 optimistically — if rejected, the API clamps to its max.
                // Multi-pass identification in geminiNanoService.ts is the robust solution
                // regardless of this value.
                requestBuilder.maxOutputTokens = 1024
                requestBuilder.topK = 10
                val request = requestBuilder.build()

                val response = withContext(Dispatchers.Main) { model.generateContent(request) }
                val text = response.candidates.firstOrNull()?.text
                if (text.isNullOrEmpty()) {
                    "ERROR:empty_response candidates=${response.candidates.size} finishReason=${response.candidates.firstOrNull()?.finishReason}"
                } else {
                    text
                }
            } catch (e: com.google.mlkit.genai.common.GenAiException) {
                "ERROR:GenAiException:code=${e.errorCode}:${e.message}"
            } catch (e: Exception) {
                "ERROR:${e.javaClass.simpleName}:${e.message}"
            }
        }
    }

    /**
     * Scale bitmap down so the longest edge is at most maxPx.
     * Always returns a software ARGB_8888 bitmap — createScaledBitmap can produce
     * hardware-config bitmaps on newer Android, which AICore CPU inference can't read.
     */
    private fun scaleBitmapIfNeeded(bitmap: Bitmap, maxPx: Int): Bitmap {
        val maxEdge = maxOf(bitmap.width, bitmap.height)
        val (targetW, targetH) = if (maxEdge <= maxPx) {
            bitmap.width to bitmap.height
        } else {
            val scale = maxPx.toFloat() / maxEdge
            (bitmap.width * scale).toInt() to (bitmap.height * scale).toInt()
        }
        // Copy via Canvas to guarantee software ARGB_8888 regardless of source config.
        val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawBitmap(bitmap, null, android.graphics.RectF(0f, 0f, targetW.toFloat(), targetH.toFloat()), null)
        return out
    }
}
