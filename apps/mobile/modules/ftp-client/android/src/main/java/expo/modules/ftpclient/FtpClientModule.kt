package expo.modules.ftpclient

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.apache.commons.net.ftp.FTP
import org.apache.commons.net.ftp.FTPClient
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

class FtpClientModule : Module() {
  private val scope = CoroutineScope(Dispatchers.IO)

  override fun definition() = ModuleDefinition {
    Name("FtpClient")

    AsyncFunction("upload") { host: String, port: Int, user: String, pass: String, remotePath: String, localPath: String, promise: Promise ->
      scope.launch {
        val client = FTPClient()
        try {
          client.connectTimeout = 15_000
          client.defaultTimeout = 30_000
          client.connect(host, port)
          client.login(user, pass)
          client.enterLocalPassiveMode()
          client.setFileType(FTP.BINARY_FILE_TYPE)

          // Ensure remote directory exists
          val remoteDir = remotePath.substringBeforeLast('/')
          if (remoteDir.isNotEmpty()) {
            client.makeDirectory(remoteDir)
          }

          val localFile = File(localPath.removePrefix("file://"))
          FileInputStream(localFile).use { fis ->
            val success = client.storeFile(remotePath, fis)
            if (!success) {
              promise.reject("FTP_UPLOAD_FAILED", "FTP storeFile returned false: ${client.replyString}", null)
              return@launch
            }
          }

          promise.resolve(null)
        } catch (e: Exception) {
          promise.reject("FTP_UPLOAD_ERROR", e.message ?: "Unknown FTP error", e)
        } finally {
          try { client.logout() } catch (_: Exception) {}
          try { client.disconnect() } catch (_: Exception) {}
        }
      }
    }

    AsyncFunction("download") { host: String, port: Int, user: String, pass: String, remotePath: String, localPath: String, promise: Promise ->
      scope.launch {
        val client = FTPClient()
        try {
          client.connectTimeout = 15_000
          client.defaultTimeout = 30_000
          client.connect(host, port)
          client.login(user, pass)
          client.enterLocalPassiveMode()
          client.setFileType(FTP.BINARY_FILE_TYPE)

          val localFile = File(localPath.removePrefix("file://"))
          localFile.parentFile?.mkdirs()

          FileOutputStream(localFile).use { fos ->
            val success = client.retrieveFile(remotePath, fos)
            if (!success) {
              promise.reject("FTP_DOWNLOAD_FAILED", "FTP retrieveFile returned false: ${client.replyString}", null)
              return@launch
            }
          }

          promise.resolve(null)
        } catch (e: Exception) {
          promise.reject("FTP_DOWNLOAD_ERROR", e.message ?: "Unknown FTP error", e)
        } finally {
          try { client.logout() } catch (_: Exception) {}
          try { client.disconnect() } catch (_: Exception) {}
        }
      }
    }

    AsyncFunction("testConnection") { host: String, port: Int, user: String, pass: String, promise: Promise ->
      scope.launch {
        val client = FTPClient()
        try {
          client.connectTimeout = 10_000
          client.defaultTimeout = 10_000
          client.connect(host, port)
          val loginOk = client.login(user, pass)
          if (!loginOk) {
            promise.resolve(false)
            return@launch
          }
          client.enterLocalPassiveMode()
          // Quick list to verify connection works
          client.listFiles("/")
          promise.resolve(true)
        } catch (_: Exception) {
          promise.resolve(false)
        } finally {
          try { client.logout() } catch (_: Exception) {}
          try { client.disconnect() } catch (_: Exception) {}
        }
      }
    }
  }
}
