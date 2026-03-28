package expo.modules.geminanano

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.TextPart
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * BroadcastReceiver for DSPy prompt evaluation.
 *
 * Trigger via adb:
 *   adb shell am broadcast -a com.foodtracker.IDENTIFY_FOOD \
 *     --es image_path "/sdcard/foodtracker_eval/input.jpg" \
 *     --es prompt_path "/sdcard/foodtracker_eval/prompt.txt" \
 *     --es result_path "/sdcard/foodtracker_eval/result.txt" \
 *     com.jtingexpo.mobile/expo.modules.geminanano.VlmEvalReceiver
 *
 * prompt_path: file containing the prompt text (avoids shell escaping issues).
 * Falls back to "prompt" string extra if prompt_path is not provided.
 *
 * The receiver writes the raw Gemini Nano text output (or ERROR:...) to result_path,
 * then writes a sentinel file at result_path + ".done" to signal completion.
 */
class VlmEvalReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        // Use app's external files dir — accessible by both adb and the app on Android 11+
        val evalDir = File(context.getExternalFilesDir(null), "eval")
        evalDir.mkdirs()

        val imagePath = intent.getStringExtra("image_path")
            ?: File(evalDir, "input.jpg").absolutePath
        // Read prompt from file (preferred) or inline string extra
        val promptPath = intent.getStringExtra("prompt_path")
            ?: File(evalDir, "prompt.txt").absolutePath
        val prompt = run {
            val f = File(promptPath)
            if (f.exists()) {
                f.readText()
            } else {
                intent.getStringExtra("prompt") ?: run {
                    writeResult(intent, context, "ERROR:missing_prompt")
                    return
                }
            }
        }
        val resultPath = intent.getStringExtra("result_path")
            ?: File(evalDir, "result.txt").absolutePath

        // goAsync() extends the receiver's lifetime from 10s to ~30s
        val pendingResult = goAsync()

        CoroutineScope(Dispatchers.Main).launch {
            try {
                val result = runInference(imagePath, prompt)
                writeToFile(resultPath, result)
            } catch (e: Exception) {
                writeToFile(resultPath, "ERROR:${e.javaClass.simpleName}:${e.message}")
            } finally {
                pendingResult.finish()
            }
        }
    }

    private suspend fun runInference(imagePath: String, prompt: String): String {
        val imageFile = File(imagePath)
        if (!imageFile.exists()) return "ERROR:image_not_found:$imagePath"

        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val rawBitmap = BitmapFactory.decodeFile(imagePath, opts)
            ?: return "ERROR:decode_failed:null bitmap"
        val bitmap = scaleBitmapIfNeeded(rawBitmap, 512)

        val model = Generation.getClient()
        val requestBuilder = GenerateContentRequest.Builder(ImagePart(bitmap), TextPart(prompt))
        requestBuilder.temperature = 0.2f
        requestBuilder.maxOutputTokens = 256
        requestBuilder.topK = 10
        val request = requestBuilder.build()

        val response = withContext(Dispatchers.Main) { model.generateContent(request) }
        val text = response.candidates.firstOrNull()?.text
        return if (text.isNullOrEmpty()) {
            "ERROR:empty_response:candidates=${response.candidates.size}"
        } else {
            text
        }
    }

    private fun scaleBitmapIfNeeded(bitmap: Bitmap, maxPx: Int): Bitmap {
        val maxEdge = maxOf(bitmap.width, bitmap.height)
        if (maxEdge <= maxPx) return bitmap
        val scale = maxPx.toFloat() / maxEdge
        val targetW = (bitmap.width * scale).toInt()
        val targetH = (bitmap.height * scale).toInt()
        val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawBitmap(bitmap, null, android.graphics.RectF(0f, 0f, targetW.toFloat(), targetH.toFloat()), null)
        return out
    }

    private fun writeResult(intent: Intent, context: Context, error: String) {
        val evalDir = File(context.getExternalFilesDir(null), "eval")
        val resultPath = intent.getStringExtra("result_path")
            ?: File(evalDir, "result.txt").absolutePath
        writeToFile(resultPath, error)
    }

    private fun writeToFile(resultPath: String, content: String) {
        val resultFile = File(resultPath)
        resultFile.parentFile?.mkdirs()
        resultFile.writeText(content)
        // Sentinel file signals completion to the polling Python side
        File("$resultPath.done").writeText("ok")
    }
}
