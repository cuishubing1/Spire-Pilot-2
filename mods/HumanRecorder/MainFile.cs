using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;

namespace Sts2HumanRecorder;

[ModInitializer(nameof(Initialize))]
public static class MainFile
{
    public const string ModId = "HumanRecorder";
    public const string RecorderVersion = "0.5.3-internal";
    public const string LiveSchemaVersion = "human-live-0.4.1";
    public static MegaCrit.Sts2.Core.Logging.Logger Logger { get; } =
        new(ModId, MegaCrit.Sts2.Core.Logging.LogType.Generic);

    public static void Initialize()
    {
        try
        {
            RecorderSession.AssertCompatibleGame();
            var harmony = new Harmony(ModId);
            PatchRegistry.Install(harmony);
            RecorderSession.Initialize();
            try { RecorderOverlay.Install(); }
            catch (Exception overlayError) { Logger.Warn("HumanRecorder overlay unavailable: " + overlayError); }
            Logger.Info("HumanRecorder 0.5.3-internal initialized; gameplay is not modified.");
        }
        catch (Exception ex)
        {
            Logger.Error($"HumanRecorder initialization failed: {ex}");
            RecorderSession.ReportFatalInitialization(ex);
        }
    }
}
