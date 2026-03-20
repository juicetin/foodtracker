package expo.modules.aipackdelivery

import com.google.android.play.core.aipack.AiPackManager
import com.google.android.play.core.aipack.AiPackManagerFactory
import com.google.android.play.core.aipack.AiPackRequest
import com.google.android.play.core.aipack.AiPackState
import com.google.android.play.core.aipack.model.AiPackStatus
import expo.modules.kotlin.functions.Coroutine
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.tasks.await

class AiPackDeliveryModule : Module() {

    private var manager: AiPackManager? = null

    private fun getManager(): AiPackManager {
        if (manager == null) {
            manager = AiPackManagerFactory.create(appContext.reactContext!!)
        }
        return manager!!
    }

    override fun definition() = ModuleDefinition {
        Name("AiPackDelivery")

        /**
         * Get the download/install status of an AI pack.
         * Returns: "completed" | "pending" | "downloading" | "not_installed" | "unknown"
         */
        AsyncFunction("getPackStatus") Coroutine { packName: String ->
            try {
                val states = getManager().getPackStates(listOf(packName)).await()
                val packState: AiPackState? = states.packStates()[packName]
                when (packState?.status()) {
                    AiPackStatus.COMPLETED -> "completed"
                    AiPackStatus.PENDING -> "pending"
                    AiPackStatus.DOWNLOADING -> "downloading"
                    AiPackStatus.NOT_INSTALLED -> "not_installed"
                    else -> "unknown"
                }
            } catch (e: Exception) {
                "unknown"
            }
        }

        /**
         * Get the local filesystem path for a completed AI pack's assets.
         * Returns the assets path if pack is completed, null otherwise.
         */
        AsyncFunction("getPackLocation") Coroutine { packName: String ->
            try {
                val states = getManager().getPackStates(listOf(packName)).await()
                val packState: AiPackState? = states.packStates()[packName]
                if (packState?.status() == AiPackStatus.COMPLETED) {
                    val location = getManager().getPackLocation(packName)
                    location?.assetsPath()
                } else {
                    null
                }
            } catch (e: Exception) {
                null
            }
        }

        /**
         * Request download of an AI pack.
         * Returns true if the request was successfully submitted, false on error.
         */
        AsyncFunction("requestDownload") Coroutine { packName: String ->
            try {
                val request = AiPackRequest.newBuilder()
                    .addPack(packName)
                    .build()
                getManager().fetch(request).await()
                true
            } catch (e: Exception) {
                false
            }
        }
    }
}
