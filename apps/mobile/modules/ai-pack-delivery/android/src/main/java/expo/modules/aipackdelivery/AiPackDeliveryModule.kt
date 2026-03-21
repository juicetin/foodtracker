package expo.modules.aipackdelivery

import com.google.android.play.core.aipacks.AiPackManager
import com.google.android.play.core.aipacks.AiPackManagerFactory
import com.google.android.play.core.aipacks.AiPackState
import com.google.android.play.core.aipacks.AiPackStates
import com.google.android.play.core.aipacks.model.AiPackStatus
import expo.modules.kotlin.functions.Coroutine
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.tasks.await

class AiPackDeliveryModule : Module() {

    private var manager: AiPackManager? = null

    private fun getManager(): AiPackManager {
        if (manager == null) {
            manager = AiPackManagerFactory.getInstance(appContext.reactContext!!)
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
                val states: AiPackStates = getManager().getPackStates(listOf(packName)).await()
                val stateMap: Map<String, AiPackState> = states.packStates()
                val packState: AiPackState? = stateMap[packName]
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
                val states: AiPackStates = getManager().getPackStates(listOf(packName)).await()
                val stateMap: Map<String, AiPackState> = states.packStates()
                val packState: AiPackState? = stateMap[packName]
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
                getManager().fetch(listOf(packName)).await()
                true
            } catch (e: Exception) {
                false
            }
        }
    }
}
