import ExpoModulesCore

/// No-op stub for iOS. Play for On-Device AI is Android-only.
/// iOS On-Demand Resources deferred per user decision (MDL-02).
public class AiPackDeliveryModule: Module {
    public func definition() -> ModuleDefinition {
        Name("AiPackDelivery")

        AsyncFunction("getPackStatus") { (_: String) -> String in
            return "unknown"
        }

        AsyncFunction("getPackLocation") { (_: String) -> String? in
            return nil
        }

        AsyncFunction("requestDownload") { (_: String) -> Bool in
            return false
        }
    }
}
