import ExpoModulesCore

public class FtpClientModule: Module {
  public func definition() -> ModuleDefinition {
    Name("FtpClient")

    AsyncFunction("upload") { (_: String, _: Int, _: String, _: String, _: String, _: String) -> Void in
      throw Exception(name: "FTP_NOT_AVAILABLE", description: "FTP is not available on iOS yet")
    }

    AsyncFunction("download") { (_: String, _: Int, _: String, _: String, _: String, _: String) -> Void in
      throw Exception(name: "FTP_NOT_AVAILABLE", description: "FTP is not available on iOS yet")
    }

    AsyncFunction("testConnection") { (_: String, _: Int, _: String, _: String) -> Bool in
      throw Exception(name: "FTP_NOT_AVAILABLE", description: "FTP is not available on iOS yet")
    }
  }
}
